#!/usr/bin/env python3
"""
실전용 IMU 기반 경로 난이도 추정 파이프라인
두 데이터셋 역할 분담: HAR-PMD(기초 표현) + Stairclimbing(난이도 시드)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class ProductionIMUPipeline:
    """실전용 IMU 파이프라인"""

    def __init__(self, sampling_rate=50, window_seconds=2.0, overlap_ratio=0.5):
        self.sampling_rate = sampling_rate
        self.window_size = int(window_seconds * sampling_rate)  # 100 samples
        self.stride = int(self.window_size * (1 - overlap_ratio))  # 50 samples

        # 전처리 컴포넌트
        self.scaler = RobustScaler()
        self.pca = None

        # 모델 컴포넌트
        self.base_model = None
        self.difficulty_mapper = None

        # 라벨 매핑
        self.difficulty_map = {
            0: "평지 (Easy)",
            1: "경사 (Moderate)",
            2: "계단 (Hard)",
            3: "급경사/장애물 (Extreme)"
        }

        # 엣지 비용 계수
        self.alpha = 1.0  # 거리 가중치
        self.beta = 2.0   # 난이도 가중치
        self.gamma = 10.0 # 제약 패널티

    def preprocess_imu_stream(self, df, gravity_compensation=True):
        """IMU 스트림 전처리"""
        print("🔧 IMU 데이터 전처리...")

        # 1. 50Hz 리샘플링 (필요시)
        if len(df) > 0:
            df = df.copy()

            # 타임스탬프가 있다면 활용
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').resample(f'{1000//self.sampling_rate}ms').mean()
                df = df.dropna().reset_index()

        # 2. 센서 컬럼 확인
        acc_cols = ['ax', 'ay', 'az']
        gyro_cols = ['gx', 'gy', 'gz']

        required_cols = acc_cols + gyro_cols
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"필수 센서 컬럼 누락: {missing}")

        # 3. 중력 보정 (옵션)
        if gravity_compensation:
            df = self._gravity_compensation(df, acc_cols)

        # 4. 누락값 처리 (윈도우 드롭 방식)
        df = df.dropna(subset=required_cols)

        print(f"✅ 전처리 완료: {len(df)}개 샘플")
        return df

    def _gravity_compensation(self, df, acc_cols, cutoff_freq=0.3):
        """저역통과 필터로 중력 성분 제거"""
        sos = signal.butter(4, cutoff_freq, btype='low', fs=self.sampling_rate, output='sos')

        for col in acc_cols:
            # 중력 성분 추정
            gravity = signal.sosfilt(sos, df[col])
            # 중력 제거한 선형가속도
            df[f'{col}_linear'] = df[col] - gravity
            # 중력 벡터 크기도 저장
            df[f'{col}_gravity'] = gravity

        return df

    def extract_window_features(self, window_df):
        """윈도우별 특징 추출 (최소 구성)"""
        acc_cols = ['ax', 'ay', 'az']
        gyro_cols = ['gx', 'gy', 'gz']

        features = {}

        # 가속도 벡터 크기
        acc_mag = np.sqrt(window_df[acc_cols[0]]**2 +
                         window_df[acc_cols[1]]**2 +
                         window_df[acc_cols[2]]**2)

        # 자이로 벡터 크기
        gyro_mag = np.sqrt(window_df[gyro_cols[0]]**2 +
                          window_df[gyro_cols[1]]**2 +
                          window_df[gyro_cols[2]]**2)

        # === 시간영역 특성 ===
        features.update({
            'mean_acc': np.mean(acc_mag),
            'var_acc': np.var(acc_mag),
            'rms': np.sqrt(np.mean(acc_mag**2)),
            'sma': np.mean(np.abs(acc_mag)),
            'mean_gyro': np.mean(gyro_mag),
            'var_gyro': np.var(gyro_mag),
        })

        # Jerk (평활 후 1차 차분)
        acc_smooth = signal.savgol_filter(acc_mag, 5, 2)
        jerk = np.sum(np.abs(np.diff(acc_smooth)))
        features['jerk'] = jerk

        # Peak count (prominence 기준)
        peaks, properties = signal.find_peaks(acc_mag, prominence=0.5)
        features['peak_count'] = len(peaks)

        # Zero crossing rate
        zcr = np.sum(np.diff(np.sign(acc_mag - np.mean(acc_mag))) != 0)
        features['zcr'] = zcr

        # === 주파수영역 특성 ===
        freqs, psd = signal.welch(acc_mag, fs=self.sampling_rate, nperseg=min(64, len(acc_mag)))

        # 0.5-5Hz 밴드파워
        band_mask = (freqs >= 0.5) & (freqs <= 5.0)
        band_power = np.sum(psd[band_mask])
        features['band_power'] = band_power

        # 지배 주파수
        dom_freq = freqs[np.argmax(psd)]
        features['dom_freq'] = dom_freq

        # 스펙트럴 엔트로피
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        features['spectral_entropy'] = spectral_entropy

        # === 자세/경사 프록시 ===
        if all(f'{col}_gravity' in window_df.columns for col in acc_cols):
            # 중력벡터와의 각도
            gravity_vec = np.column_stack([window_df[f'{col}_gravity'] for col in acc_cols])
            gravity_angles = np.arccos(np.clip(gravity_vec[:, 2] /
                                             np.linalg.norm(gravity_vec, axis=1), -1, 1))
            features['gravity_angle_mean'] = np.mean(gravity_angles)
            features['gravity_angle_var'] = np.var(gravity_angles)

        return features

    def create_sliding_windows(self, df):
        """슬라이딩 윈도우 생성"""
        windows = []
        window_positions = []

        for start in range(0, len(df) - self.window_size + 1, self.stride):
            end = start + self.window_size
            window_df = df.iloc[start:end].copy()

            # 윈도우 내 누락값이 있으면 스킵
            if window_df.isnull().any().any():
                continue

            windows.append(window_df)
            window_positions.append((start, end))

        return windows, window_positions

    def offline_fit(self, har_data_path=None, stair_data_path="data/raw/combined_dataset_stairclimbing.csv"):
        """오프라인 학습 (HAR-PMD + Stairclimbing)"""
        print("🎯 오프라인 모델 학습 시작")
        print("="*50)

        # === 1단계: Stairclimbing 데이터로 시드 라벨 생성 ===
        print("\n1️⃣ Stairclimbing 데이터 처리...")
        stair_df = pd.read_csv(stair_data_path).dropna()

        # 전처리
        stair_df = self.preprocess_imu_stream(stair_df)

        # 윈도우 생성
        stair_windows, _ = self.create_sliding_windows(stair_df)
        print(f"   생성된 윈도우: {len(stair_windows)}개")

        # 특징 추출
        stair_features = []
        stair_labels = []

        for window_df in stair_windows:
            features = self.extract_window_features(window_df)
            label = window_df['label'].mode()[0]  # 윈도우 내 최빈 라벨

            stair_features.append(list(features.values()))
            stair_labels.append(label)

        X_stair = np.array(stair_features)

        print(f"   추출된 특징: {X_stair.shape}")
        print(f"   라벨 분포: {pd.Series(stair_labels).value_counts().to_dict()}")

        # === 2단계: 특징 전처리 ===
        print("\n2️⃣ 특징 전처리...")
        X_stair_scaled = self.scaler.fit_transform(X_stair)

        # PCA (옵션)
        self.pca = PCA(n_components=0.95)  # 95% 분산 보존
        X_stair_pca = self.pca.fit_transform(X_stair_scaled)

        print(f"   PCA 차원: {X_stair.shape[1]} → {X_stair_pca.shape[1]}")

        # === 3단계: 난이도 시드 라벨 매핑 ===
        print("\n3️⃣ 난이도 시드 라벨 매핑...")
        difficulty_labels = self._map_to_difficulty(stair_labels)

        # === 4단계: 기초 모델 학습 ===
        print("\n4️⃣ 기초 분류 모델 학습...")
        self.base_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )

        self.base_model.fit(X_stair_pca, difficulty_labels)

        # 교차검증 평가
        cv_scores = []
        cv = LeaveOneGroupOut()
        groups = [label.split('_')[0] for label in stair_labels]  # 활동 유형별 그룹

        for train_idx, test_idx in cv.split(X_stair_pca, difficulty_labels, groups):
            X_train, X_test = X_stair_pca[train_idx], X_stair_pca[test_idx]
            y_train, y_test = np.array(difficulty_labels)[train_idx], np.array(difficulty_labels)[test_idx]

            temp_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            temp_model.fit(X_train, y_train)

            y_pred = temp_model.predict(X_test)
            score = f1_score(y_test, y_pred, average='weighted')
            cv_scores.append(score)

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        print(f"   교차검증 F1 점수: {cv_mean:.3f} ± {cv_std:.3f}")

        # === 5단계: 아티팩트 저장 ===
        print("\n5️⃣ 모델 아티팩트 저장...")
        self._save_artifacts()

        print(f"\n✅ 오프라인 학습 완료!")
        return cv_mean

    def _map_to_difficulty(self, activity_labels):
        """활동 라벨을 난이도로 매핑"""
        activity_to_difficulty = {
            'up_stairs': 2,       # 계단 올라가기
            'down_stairs': 2,     # 계단 내려가기
            'up_largestairs': 3,  # 큰 계단 올라가기
            'down_largestair': 3, # 큰 계단 내려가기
            'up_slope': 1,        # 경사 올라가기
            'down_slope': 1,      # 경사 내려가기
        }

        return [activity_to_difficulty.get(label, 0) for label in activity_labels]

    def online_infer(self, stream_df):
        """온라인 추론 (실시간 스트림)"""
        print("🔮 온라인 난이도 추론...")

        # 전처리
        stream_df = self.preprocess_imu_stream(stream_df)

        # 윈도우 생성
        windows, positions = self.create_sliding_windows(stream_df)

        if not windows:
            print("❌ 유효한 윈도우가 없습니다.")
            return None

        # 특징 추출
        features_list = []
        for window_df in windows:
            features = self.extract_window_features(window_df)
            features_list.append(list(features.values()))

        X = np.array(features_list)

        # 전처리 변환
        X_scaled = self.scaler.transform(X)

        # PCA가 없으면 스케일된 데이터 그대로 사용
        if self.pca is not None:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled

        # 난이도 예측
        difficulty_preds = self.base_model.predict(X_processed)
        difficulty_proba = self.base_model.predict_proba(X_processed)

        # 연속성 보정 (스무딩)
        difficulty_smoothed = self._smooth_predictions(difficulty_preds)

        # 엣지 비용 계산
        edge_costs = self._calculate_edge_costs(difficulty_smoothed, positions)

        results = {
            'positions': positions,
            'difficulty_raw': difficulty_preds,
            'difficulty_smoothed': difficulty_smoothed,
            'probabilities': difficulty_proba,
            'edge_costs': edge_costs
        }

        print(f"✅ {len(windows)}개 세그먼트 처리 완료")
        return results

    def _smooth_predictions(self, predictions, min_duration=3):
        """세그먼트 스무딩 (최소 지속시간 적용)"""
        if len(predictions) < min_duration:
            return predictions

        # 양방향 이동평균
        smoothed = signal.medfilt(predictions, kernel_size=min(min_duration, len(predictions)))

        return smoothed.astype(int)

    def _calculate_edge_costs(self, difficulties, positions):
        """난이도를 엣지 비용으로 변환"""
        costs = []

        for i in range(len(difficulties)):
            # 기본 비용 매핑
            difficulty_cost = difficulties[i] * 2  # 0,2,4,6

            # 거리 성분 (윈도우 길이 기준)
            if i < len(positions):
                start, end = positions[i]
                distance_cost = self.alpha * (end - start)
            else:
                distance_cost = self.alpha * self.stride

            # 총 비용
            total_cost = distance_cost + self.beta * difficulty_cost
            costs.append(total_cost)

        return costs

    def _save_artifacts(self):
        """모델 아티팩트 저장"""
        os.makedirs('models', exist_ok=True)

        artifacts = {
            'scaler': self.scaler,
            'pca': self.pca,
            'base_model': self.base_model,
            'difficulty_map': self.difficulty_map,
            'sampling_rate': self.sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma
        }

        with open('models/production_pipeline.pkl', 'wb') as f:
            pickle.dump(artifacts, f)

        print("✅ 아티팩트 저장: models/production_pipeline.pkl")

    def load_artifacts(self, path='models/production_pipeline.pkl'):
        """저장된 아티팩트 로드"""
        with open(path, 'rb') as f:
            artifacts = pickle.load(f)

        self.scaler = artifacts['scaler']
        self.pca = artifacts['pca']
        self.base_model = artifacts['base_model']
        self.difficulty_map = artifacts['difficulty_map']

        print("✅ 아티팩트 로드 완료")

def main():
    """메인 실행"""
    print("🚀 실전용 IMU 파이프라인")
    print("="*50)

    # 파이프라인 초기화
    pipeline = ProductionIMUPipeline()

    # 오프라인 학습
    cv_score = pipeline.offline_fit()

    print(f"\n🎉 파이프라인 구축 완료!")
    print(f"📊 모델 성능: F1={cv_score:.3f}")

if __name__ == "__main__":
    main()