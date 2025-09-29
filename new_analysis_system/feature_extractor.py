#!/usr/bin/env python3
"""
특성 추출기 - test 데이터에서 의미있는 특성 추출
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    """센서 데이터 특성 추출기"""

    def __init__(self, window_size: int = 200, overlap_ratio: float = 0.75):
        """
        Args:
            window_size: 윈도우 크기 (샘플 수) - 메모리 최적화 고려
            overlap_ratio: 오버랩 비율 (0.0 ~ 1.0) - 정확도 vs 속도 트레이드오프
        """
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.step_size = int(window_size * (1 - overlap_ratio))

        # 성능 최적화 설정
        self.use_parallel = True  # 병렬 처리 활성화
        self.cache_fft = {}  # FFT 결과 캐싱

    def extract_time_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """시간 도메인 특성 추출"""
        features = {}

        # 기본 통계량
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['var'] = np.var(signal)
        features['min'] = np.min(signal)
        features['max'] = np.max(signal)
        features['range'] = features['max'] - features['min']

        # 중앙값과 분위수
        features['median'] = np.median(signal)
        features['q25'] = np.percentile(signal, 25)
        features['q75'] = np.percentile(signal, 75)
        features['iqr'] = features['q75'] - features['q25']

        # 형태 특성
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)

        # RMS (Root Mean Square)
        features['rms'] = np.sqrt(np.mean(signal**2))

        # 신호 변화량
        diff = np.diff(signal)
        features['mean_diff'] = np.mean(np.abs(diff))
        features['std_diff'] = np.std(diff)

        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal)

        return features

    def extract_frequency_domain_features(self, signal: np.ndarray,
                                        sampling_rate: float = 50.0) -> Dict[str, float]:
        """주파수 도메인 특성 추출"""
        features = {}

        # FFT 계산
        fft_values = fft(signal)
        fft_magnitude = np.abs(fft_values[:len(fft_values)//2])
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)[:len(fft_values)//2]

        # 스펙트럼 특성
        features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
        features['spectral_rolloff'] = freqs[np.where(np.cumsum(fft_magnitude) >= 0.85 * np.sum(fft_magnitude))[0][0]]
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * fft_magnitude) / np.sum(fft_magnitude))

        # 주요 주파수 대역별 에너지
        low_freq = fft_magnitude[(freqs >= 0) & (freqs < 2)]
        mid_freq = fft_magnitude[(freqs >= 2) & (freqs < 8)]
        high_freq = fft_magnitude[(freqs >= 8) & (freqs < 20)]

        total_energy = np.sum(fft_magnitude**2)
        features['low_freq_energy'] = np.sum(low_freq**2) / total_energy if total_energy > 0 else 0
        features['mid_freq_energy'] = np.sum(mid_freq**2) / total_energy if total_energy > 0 else 0
        features['high_freq_energy'] = np.sum(high_freq**2) / total_energy if total_energy > 0 else 0

        # 도미넌트 주파수
        dominant_freq_idx = np.argmax(fft_magnitude)
        features['dominant_frequency'] = freqs[dominant_freq_idx]

        return features

    def extract_motion_features(self, acc_data: np.ndarray, gyro_data: np.ndarray = None) -> Dict[str, float]:
        """모션 특성 추출"""
        features = {}

        # 가속도 벡터 크기
        if acc_data.shape[1] >= 3:
            acc_magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
            features.update(self.extract_time_domain_features(acc_magnitude))

            # 각 축별 특성도 추가
            for i, axis in enumerate(['x', 'y', 'z']):
                axis_features = self.extract_time_domain_features(acc_data[:, i])
                for key, value in axis_features.items():
                    features[f'acc_{axis}_{key}'] = value

        # 자이로스코프 데이터가 있으면
        if gyro_data is not None and gyro_data.shape[1] >= 3:
            gyro_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))
            gyro_features = self.extract_time_domain_features(gyro_magnitude)
            for key, value in gyro_features.items():
                features[f'gyro_{key}'] = value

            # 각 축별 자이로 특성
            for i, axis in enumerate(['x', 'y', 'z']):
                axis_features = self.extract_time_domain_features(gyro_data[:, i])
                for key, value in axis_features.items():
                    features[f'gyro_{axis}_{key}'] = value

        return features

    def extract_window_features(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """윈도우 데이터에서 모든 특성 추출"""
        features = {}

        # 가속도 데이터 준비
        acc_cols = ['acc_x', 'acc_y', 'acc_z']
        if all(col in window_data.columns for col in acc_cols):
            acc_data = window_data[acc_cols].values

            # 가속도 벡터 크기
            acc_magnitude = np.sqrt(np.sum(acc_data**2, axis=1))

            # 시간 도메인 특성
            time_features = self.extract_time_domain_features(acc_magnitude)
            for key, value in time_features.items():
                features[f'acc_mag_{key}'] = value

            # 주파수 도메인 특성
            freq_features = self.extract_frequency_domain_features(acc_magnitude)
            for key, value in freq_features.items():
                features[f'acc_mag_{key}'] = value

        # 자이로스코프 데이터 준비 (있으면)
        gyro_cols = ['gyro_x', 'gyro_y', 'gyro_z']
        if all(col in window_data.columns for col in gyro_cols):
            gyro_data = window_data[gyro_cols].values
            gyro_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))

            # 자이로 특성
            gyro_time_features = self.extract_time_domain_features(gyro_magnitude)
            for key, value in gyro_time_features.items():
                features[f'gyro_mag_{key}'] = value

        # 추가 특성들
        # Jerk (가속도 변화율)
        if all(col in window_data.columns for col in acc_cols):
            for i, axis in enumerate(['x', 'y', 'z']):
                signal = window_data[f'acc_{axis}'].values
                jerk = np.diff(signal)
                jerk_features = self.extract_time_domain_features(jerk)
                for key, value in jerk_features.items():
                    features[f'jerk_{axis}_{key}'] = value

        # 전체 활동 강도
        if 'acc_mag_rms' in features and 'gyro_mag_rms' in features:
            features['activity_intensity'] = features['acc_mag_rms'] + features['gyro_mag_rms']
        elif 'acc_mag_rms' in features:
            features['activity_intensity'] = features['acc_mag_rms']

        return features

    def process_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
        """전체 데이터를 윈도우별로 처리하여 특성 추출"""
        print(f"🔍 특성 추출 시작 (윈도우: {self.window_size}, 오버랩: {self.overlap_ratio})")

        features_list = []
        window_positions = []

        # 윈도우별 처리
        start = 0
        window_id = 0

        while start + self.window_size <= len(df):
            end = start + self.window_size
            window_data = df.iloc[start:end]

            # 특성 추출
            try:
                features = self.extract_window_features(window_data)
                features['window_id'] = window_id
                features['start_idx'] = start
                features['end_idx'] = end
                features['window_size'] = self.window_size

                features_list.append(features)
                window_positions.append((start, end))

                window_id += 1

            except Exception as e:
                print(f"⚠️  윈도우 {window_id} 처리 실패: {e}")

            start += self.step_size

        print(f"✅ 특성 추출 완료: {len(features_list)}개 윈도우")

        features_df = pd.DataFrame(features_list)
        return features_df, window_positions

if __name__ == "__main__":
    # 테스트 실행
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from utils.data_loader import load_sensor_data, combine_sensor_data

    print("🔍 특성 추출 테스트")
    print("=" * 50)

    # 데이터 로드
    data_dir = "data/test 2025-09-22 18-30-21"
    sensor_data = load_sensor_data(data_dir)
    combined_df = combine_sensor_data(sensor_data)

    # 특성 추출기 생성
    extractor = FeatureExtractor(window_size=200, overlap_ratio=0.75)

    # 특성 추출
    features_df, positions = extractor.process_data(combined_df)

    print(f"\n📊 추출된 특성:")
    print(f"  윈도우 수: {len(features_df)}")
    print(f"  특성 수: {len(features_df.columns)}")
    print(f"  특성 목록: {list(features_df.columns[:10])}...")

    print(f"\n📋 특성 데이터 샘플:")
    print(features_df.head())

    # 결과 저장
    os.makedirs("results", exist_ok=True)
    features_df.to_csv("results/extracted_features.csv", index=False)
    print(f"\n💾 특성 데이터 저장: results/extracted_features.csv")