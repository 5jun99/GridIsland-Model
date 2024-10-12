#!/usr/bin/env python3
"""
향상된 GPS 데이터 로더 - 기존 feature_extractor와 clustering_analyzer, difficulty_analyzer 통합
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

# 기존 모듈들 import
from feature_extractor import FeatureExtractor
from clustering_analyzer import ClusteringAnalyzer
from difficulty_analyzer import DifficultyAnalyzer

class EnhancedGPSLoader:
    """기존 분석 시스템과 통합된 GPS 데이터 로더"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.gps_data = None
        self.sensor_data = {}
        self.synchronized_data = None
        self.extracted_features = None
        self.clustered_data = None
        self.difficulty_results = None
        
        # 기존 분석기들 초기화
        self.feature_extractor = FeatureExtractor()
        self.clustering_analyzer = ClusteringAnalyzer()
        self.difficulty_analyzer = DifficultyAnalyzer()
    
    def load_gps_data(self) -> pd.DataFrame:
        """GPS 위치 데이터 로드"""
        gps_file = self.data_path / "Location.csv"
        
        if not gps_file.exists():
            raise FileNotFoundError(f"GPS 데이터 파일을 찾을 수 없습니다: {gps_file}")
        
        print(f"📍 GPS 데이터 로드: {gps_file}")
        
        self.gps_data = pd.read_csv(gps_file)
        
        # 컬럼명 정리
        self.gps_data.columns = [
            'time_s', 'latitude', 'longitude', 'height_m', 
            'velocity_ms', 'direction_deg', 'h_accuracy_m', 'v_accuracy_deg'
        ]
        
        # NaN 값 처리
        self.gps_data['velocity_ms'] = self.gps_data['velocity_ms'].fillna(0)
        self.gps_data['direction_deg'] = self.gps_data['direction_deg'].fillna(0)
        
        print(f"✅ GPS 데이터 로드 완료: {len(self.gps_data)}개 포인트")
        
        return self.gps_data
    
    def load_sensor_data(self, sensors: List[str] = None) -> Dict[str, pd.DataFrame]:
        """센서 데이터 로드"""
        if sensors is None:
            sensors = ['Accelerometer', 'Gyroscope', 'Linear Accelerometer', 'Gravity']
        
        self.sensor_data = {}
        
        for sensor in sensors:
            sensor_file = self.data_path / f"{sensor}.csv"
            
            if sensor_file.exists():
                print(f"📱 {sensor} 데이터 로드...")
                
                df = pd.read_csv(sensor_file)
                
                # 컬럼명 정리 (원본 컬럼명 확인)
                original_columns = df.columns.tolist()
                print(f"     원본 컬럼명: {original_columns}")
                
                if sensor == 'Accelerometer':
                    df.columns = ['time_s', 'acc_x', 'acc_y', 'acc_z']
                elif sensor == 'Gyroscope':
                    df.columns = ['time_s', 'gyro_x', 'gyro_y', 'gyro_z']
                elif sensor == 'Linear Accelerometer':
                    df.columns = ['time_s', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z']
                elif sensor == 'Gravity':
                    df.columns = ['time_s', 'grav_x', 'grav_y', 'grav_z']
                
                self.sensor_data[sensor] = df
                print(f"   로드 완료: {len(df)}개 샘플")
            else:
                print(f"⚠️  {sensor} 데이터 파일 없음")
        
        return self.sensor_data
    
    def synchronize_data(self, window_size: float = 1.0) -> pd.DataFrame:
        """GPS와 센서 데이터를 시간 윈도우별로 동기화"""
        if self.gps_data is None:
            raise ValueError("GPS 데이터를 먼저 로드하세요")
        
        if not self.sensor_data:
            raise ValueError("센서 데이터를 먼저 로드하세요")
        
        print(f"🔄 데이터 동기화 시작 (윈도우 크기: {window_size}초)")
        
        # 전체 시간 범위 확인
        gps_time_range = (self.gps_data['time_s'].min(), self.gps_data['time_s'].max())
        
        sensor_time_ranges = {}
        for sensor, df in self.sensor_data.items():
            sensor_time_ranges[sensor] = (df['time_s'].min(), df['time_s'].max())
        
        # 공통 시간 범위 계산
        start_time = max([gps_time_range[0]] + [r[0] for r in sensor_time_ranges.values()])
        end_time = min([gps_time_range[1]] + [r[1] for r in sensor_time_ranges.values()])
        
        print(f"   공통 시간 범위: {start_time:.1f}s ~ {end_time:.1f}s")
        
        # 윈도우별 데이터 생성
        synchronized_records = []
        
        current_time = start_time
        window_id = 0
        
        while current_time + window_size <= end_time:
            window_end = current_time + window_size
            
            # GPS 데이터 윈도우 추출
            gps_window = self.gps_data[
                (self.gps_data['time_s'] >= current_time) & 
                (self.gps_data['time_s'] < window_end)
            ]
            
            if len(gps_window) == 0:
                current_time += window_size
                continue
            
            # GPS 핵심 정보만 계산 (위치, 속도, 고도)
            gps_stats = {
                'window_id': window_id,
                'start_time': current_time,
                'end_time': window_end,
                'duration': window_size,
                'gps_count': len(gps_window),
                'lat_mean': gps_window['latitude'].mean(),
                'lng_mean': gps_window['longitude'].mean(),
                'height_mean': gps_window['height_m'].mean(),
                'velocity_mean': gps_window['velocity_ms'].mean(),
                'direction_mean': gps_window['direction_deg'].mean()
            }
            
            synchronized_records.append(gps_stats)
            
            current_time += window_size
            window_id += 1
        
        self.synchronized_data = pd.DataFrame(synchronized_records)
        
        print(f"✅ 동기화 완료: {len(self.synchronized_data)}개 윈도우")
        
        return self.synchronized_data
    
    def extract_advanced_features(self, window_size: int = 200, overlap_ratio: float = 0.75) -> pd.DataFrame:
        """기존 feature_extractor를 사용하여 고급 특성 추출"""
        if not self.sensor_data:
            raise ValueError("센서 데이터를 먼저 로드하세요")
        
        print(f"🔍 고급 특성 추출 시작 (윈도우: {window_size}, 오버랩: {overlap_ratio})")
        
        # 센서 데이터를 DataFrame으로 통합 (기존 feature_extractor가 DataFrame을 받음)
        # 가속도계와 자이로스코프 데이터 결합
        if 'Accelerometer' in self.sensor_data and 'Gyroscope' in self.sensor_data:
            acc_data = self.sensor_data['Accelerometer']
            gyro_data = self.sensor_data['Gyroscope']
            
            print(f"   가속도계 시간 범위: {acc_data['time_s'].min():.3f} ~ {acc_data['time_s'].max():.3f}")
            print(f"   자이로스코프 시간 범위: {gyro_data['time_s'].min():.3f} ~ {gyro_data['time_s'].max():.3f}")
            
            # 시간 기준으로 병합 (tolerance 늘려서 더 많은 매칭 허용)
            merged_data = pd.merge_asof(
                acc_data.sort_values('time_s'), 
                gyro_data.sort_values('time_s'), 
                on='time_s', 
                tolerance=0.1,  # 100ms 허용 오차로 증가
                suffixes=('', '_gyro')
            )
            
            # 자이로 데이터가 제대로 병합되었는지 확인
            print(f"   자이로 데이터 유효성: {merged_data['gyro_x'].notna().sum()}/{len(merged_data)} 샘플")
            
            # 컬럼명을 기존 feature_extractor 형식에 맞게 변경
            sensor_df = merged_data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].copy()
            
            # 자이로 데이터가 없는 경우 0으로 채우기 (대신 경고 출력)
            if sensor_df['gyro_x'].isna().all():
                print("⚠️  자이로스코프 데이터 병합 실패 - 0으로 대체")
                sensor_df[['gyro_x', 'gyro_y', 'gyro_z']] = 0.0
            
            print(f"   병합된 센서 데이터 크기: {len(sensor_df)}개 샘플")
            print(f"   센서 데이터 범위: {sensor_df.index.min()} ~ {sensor_df.index.max()}")
            
            # 기존 feature_extractor로 특성 추출
            self.feature_extractor.window_size = window_size
            self.feature_extractor.overlap_ratio = overlap_ratio
            
            features_df, window_positions = self.feature_extractor.process_data(sensor_df)
            self.extracted_features = features_df
            
            print(f"✅ 특성 추출 완료: {len(self.extracted_features)}개 윈도우, {len(self.extracted_features.columns)}개 특성")
            
        else:
            raise ValueError("Accelerometer와 Gyroscope 데이터가 필요합니다")
        
        return self.extracted_features
    
    def perform_clustering(self, n_clusters: int = None) -> pd.DataFrame:
        """기존 clustering_analyzer로 클러스터링 수행"""
        if self.extracted_features is None:
            raise ValueError("특성을 먼저 추출하세요")
        
        print(f"🎯 클러스터링 분석 시작")
        
        # 특성 데이터를 파일로 임시 저장 (clustering_analyzer가 파일을 읽도록 설계됨)
        temp_features_file = "temp_features.csv"
        
        # 빈 DataFrame 체크
        if self.extracted_features is None or len(self.extracted_features) == 0:
            print("❌ 추출된 특성이 없습니다. 윈도우 크기를 줄여보세요.")
            return None
        
        self.extracted_features.to_csv(temp_features_file, index=False)
        print(f"   임시 특성 파일 저장: {len(self.extracted_features)}개 윈도우")
        
        # clustering_analyzer로 분석
        self.clustering_analyzer.load_features(temp_features_file)
        features_scaled = self.clustering_analyzer.preprocess_features()
        
        # 클러스터 수 설정 (더 세밀한 분석을 위해)
        if n_clusters is None:
            optimization_results = self.clustering_analyzer.find_optimal_clusters(max_k=10, method='kmeans')
            
            # Silhouette Score 확인하되, 최소 4개 클러스터로 설정
            best_k = optimization_results['k_values'][np.argmax(optimization_results['silhouette'])]
            
            # 4개 이상의 클러스터 강제 (더 세밀한 난이도 구분)
            if best_k < 4:
                print(f"📊 Silhouette 최적: {best_k}개 → 세밀한 분석을 위해 4개 클러스터로 조정")
                n_clusters = 4
            else:
                print(f"📊 최적 클러스터 수: {best_k} (26개 특징 기반)")
                n_clusters = best_k
        
        # 클러스터링 수행
        labels = self.clustering_analyzer.perform_clustering(n_clusters=n_clusters, method='kmeans')
        
        # 결과를 기존 특성 데이터에 추가
        self.clustered_data = self.extracted_features.copy()
        self.clustered_data['cluster'] = labels
        
        # 임시 파일 정리
        if os.path.exists(temp_features_file):
            os.remove(temp_features_file)
        
        print(f"✅ 클러스터링 완료: {n_clusters}개 클러스터")
        
        return self.clustered_data
    
    def analyze_difficulty(self) -> pd.DataFrame:
        """기존 difficulty_analyzer로 난이도 분석"""
        if self.clustered_data is None:
            raise ValueError("클러스터링을 먼저 수행하세요")
        
        print(f"🔍 난이도 분석 시작")
        
        # 클러스터별 특성 계산
        cluster_characteristics = []
        
        for cluster_id in sorted(self.clustered_data['cluster'].unique()):
            cluster_data = self.clustered_data[self.clustered_data['cluster'] == cluster_id]
            
            # 수치형 컬럼만 선택
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            exclude_cols = ['window_id', 'start_idx', 'end_idx', 'window_size', 'cluster']
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # 클러스터별 통계 계산
            cluster_stats = {
                'cluster': cluster_id,
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(self.clustered_data) * 100
            }
            
            # 각 특성의 평균값 계산
            for col in feature_cols:
                cluster_stats[f'{col}_mean'] = cluster_data[col].mean()
            
            cluster_characteristics.append(cluster_stats)
        
        cluster_characteristics_df = pd.DataFrame(cluster_characteristics)
        
        # difficulty_analyzer로 분석 (임시 파일 사용)
        temp_cluster_file = "temp_cluster_characteristics.csv"
        cluster_characteristics_df.to_csv(temp_cluster_file, index=False)
        
        self.difficulty_analyzer.load_cluster_data(temp_cluster_file)
        difficulty_results = self.difficulty_analyzer.analyze_all_clusters()
        
        self.difficulty_results = difficulty_results
        
        # 임시 파일 정리
        if os.path.exists(temp_cluster_file):
            os.remove(temp_cluster_file)
        
        print(f"✅ 난이도 분석 완료")
        
        return difficulty_results
    
    def map_difficulty_to_gps(self) -> pd.DataFrame:
        """클러스터 기반 난이도를 GPS 윈도우에 매핑"""
        if self.synchronized_data is None:
            raise ValueError("동기화된 데이터가 없습니다")
        
        if self.clustered_data is None or self.difficulty_results is None:
            raise ValueError("클러스터링과 난이도 분석을 먼저 수행하세요")
        
        print(f"🗺️  난이도를 GPS 윈도우에 매핑 중...")
        
        # GPS 윈도우와 특성 추출 윈도우 매핑
        gps_with_difficulty = self.synchronized_data.copy()
        
        # 클러스터별 난이도 정보 딕셔너리 생성
        cluster_difficulty_map = {}
        for _, row in self.difficulty_results.iterrows():
            cluster_id = row['cluster']
            cluster_difficulty_map[cluster_id] = {
                'difficulty_score': row['difficulty_score'],
                'difficulty_level': row['difficulty_level'],
                'difficulty_name': row['difficulty_name']
            }
        
        # GPS 윈도우 수와 특성 윈도우 수가 다를 수 있으므로 매핑 전략 필요
        n_gps_windows = len(self.synchronized_data)
        n_feature_windows = len(self.clustered_data)
        
        print(f"   GPS 윈도우: {n_gps_windows}개, 특성 윈도우: {n_feature_windows}개")
        
        # 특성 윈도우를 GPS 윈도우에 매핑
        difficulty_scores = []
        cluster_ids = []
        difficulty_levels = []
        
        for i in range(n_gps_windows):
            # 비례 매핑 (선형 보간)
            feature_idx = int(i * (n_feature_windows - 1) / max(1, n_gps_windows - 1))
            feature_idx = min(feature_idx, n_feature_windows - 1)
            
            cluster_id = self.clustered_data.iloc[feature_idx]['cluster']
            
            if cluster_id in cluster_difficulty_map:
                cluster_info = cluster_difficulty_map[cluster_id]
                difficulty_scores.append(cluster_info['difficulty_score'])
                difficulty_levels.append(cluster_info['difficulty_level'])
            else:
                # 기본값
                difficulty_scores.append(0.5)
                difficulty_levels.append(2)
            
            cluster_ids.append(cluster_id)
        
        # GPS 데이터에 난이도 정보 추가
        gps_with_difficulty['cluster_id'] = cluster_ids
        gps_with_difficulty['difficulty'] = difficulty_scores
        gps_with_difficulty['difficulty_level'] = difficulty_levels
        
        # 난이도 등급 문자열 추가
        def get_difficulty_name(level):
            names = ['매우 쉬움', '쉬움', '보통', '어려움', '매우 어려움']
            return names[min(level, 4)]
        
        gps_with_difficulty['difficulty_name'] = gps_with_difficulty['difficulty_level'].apply(get_difficulty_name)
        
        self.synchronized_data = gps_with_difficulty
        
        print(f"✅ 난이도 매핑 완료")
        print(f"   평균 난이도: {np.mean(difficulty_scores):.3f}")
        
        return gps_with_difficulty
    
    def visualize_enhanced_data(self, save_path: str = None):
        """향상된 데이터 시각화"""
        if self.synchronized_data is None:
            raise ValueError("매핑된 데이터가 없습니다")
        
        df = self.synchronized_data
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. GPS 경로 (클러스터별)
        ax1 = axes[0, 0]
        scatter = ax1.scatter(df['lng_mean'], df['lat_mean'], 
                             c=df['cluster_id'], cmap='tab10', 
                             s=50, alpha=0.7)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('GPS Path (colored by cluster)')
        plt.colorbar(scatter, ax=ax1)
        
        # 2. GPS 경로 (난이도별)
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['lng_mean'], df['lat_mean'], 
                             c=df['difficulty'], cmap='Reds', 
                             s=50, alpha=0.7)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('GPS Path (colored by difficulty)')
        plt.colorbar(scatter, ax=ax2)
        
        # 3. GPS 경로 (난이도 레벨별)
        ax3 = axes[0, 2]
        scatter = ax3.scatter(df['lng_mean'], df['lat_mean'], 
                             c=df['difficulty_level'], cmap='RdYlGn_r', 
                             s=50, alpha=0.7)
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.set_title('GPS Path (colored by difficulty level)')
        plt.colorbar(scatter, ax=ax3)
        
        # 4. 난이도 시계열
        ax4 = axes[1, 0]
        ax4.plot(df['window_id'], df['difficulty'], 'r-', linewidth=2)
        ax4.set_xlabel('Window ID')
        ax4.set_ylabel('Difficulty Score')
        ax4.set_title('Difficulty Over Time')
        ax4.grid(True, alpha=0.3)
        
        # 5. 난이도 레벨 시계열
        ax5 = axes[1, 1]
        ax5.plot(df['window_id'], df['difficulty_level'], 'o-', linewidth=2, markersize=4)
        ax5.set_xlabel('Window ID')
        ax5.set_ylabel('Difficulty Level')
        ax5.set_title('Difficulty Level Over Time')
        ax5.grid(True, alpha=0.3)
        
        # 6. 클러스터 분포
        ax6 = axes[1, 2]
        cluster_counts = df['cluster_id'].value_counts().sort_index()
        ax6.bar(cluster_counts.index, cluster_counts.values, alpha=0.7)
        ax6.set_xlabel('Cluster ID')
        ax6.set_ylabel('Count')
        ax6.set_title('Cluster Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 시각화 저장: {save_path}")
        
        plt.show()
    
    def save_results(self, base_path: str = "results"):
        """결과 저장"""
        os.makedirs(base_path, exist_ok=True)
        
        if self.extracted_features is not None:
            features_file = f"{base_path}/extracted_features.csv"
            self.extracted_features.to_csv(features_file, index=False)
            print(f"💾 특성 데이터 저장: {features_file}")
        
        if self.clustered_data is not None:
            clustered_file = f"{base_path}/clustered_features.csv"
            self.clustered_data.to_csv(clustered_file, index=False)
            print(f"💾 클러스터 데이터 저장: {clustered_file}")
        
        if self.difficulty_results is not None:
            difficulty_file = f"{base_path}/difficulty_analysis.csv"
            self.difficulty_results.to_csv(difficulty_file, index=False)
            print(f"💾 난이도 분석 저장: {difficulty_file}")
        
        if self.synchronized_data is not None:
            synchronized_file = f"{base_path}/gps_synchronized.csv"
            self.synchronized_data.to_csv(synchronized_file, index=False)
            print(f"💾 통합 GPS 데이터 저장: {synchronized_file}")

def main():
    """메인 실행 함수"""
    print("🚀 향상된 GPS 데이터 로더 실행")
    print("=" * 60)
    
    try:
        # 데이터 로더 초기화
        data_path = "data/Sss 2025-10-02 15-53-01"
        loader = EnhancedGPSLoader(data_path)
        
        # 1. GPS 및 센서 데이터 로드
        gps_data = loader.load_gps_data()
        sensor_data = loader.load_sensor_data()
        
        # 2. 데이터 동기화
        synchronized_data = loader.synchronize_data(window_size=1.0)
        
        # 3. 최적화된 특성 추출 (26개 핵심 특징)
        features = loader.extract_advanced_features(window_size=150, overlap_ratio=0.6)
        
        # 4. 클러스터링 (기존 clustering_analyzer 활용)
        clustered_data = loader.perform_clustering()
        
        # 5. 난이도 분석 (기존 difficulty_analyzer 활용)
        difficulty_results = loader.analyze_difficulty()
        
        # 6. GPS에 난이도 매핑
        gps_with_difficulty = loader.map_difficulty_to_gps()
        
        # 7. 시각화
        loader.visualize_enhanced_data(save_path="results/enhanced_gps_analysis.png")
        
        # 8. 결과 저장
        loader.save_results()
        
        print(f"\n✅ 향상된 GPS 분석 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()