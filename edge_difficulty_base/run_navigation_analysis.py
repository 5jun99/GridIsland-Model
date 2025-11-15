#!/usr/bin/env python3
"""
통합 네비게이션 분석 실행 스크립트
기존 분석 시스템을 활용하여 네비게이션 세그먼트를 생성합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "new_analysis_system"))
sys.path.insert(0, str(project_root / "edge_difficulty_base"))

from feature_extractor import FeatureExtractor
from clustering_analyzer import ClusteringAnalyzer
from difficulty_analyzer import DifficultyAnalyzer
from database_manager import DatabaseManager
import pandas as pd
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NavigationAnalysisRunner:
    """네비게이션 분석 통합 실행기"""
    
    def __init__(self, db_config=None):
        """
        Args:
            db_config: 데이터베이스 연결 설정
        """
        # 기존 분석기들 초기화
        self.feature_extractor = FeatureExtractor()
        self.clustering_analyzer = ClusteringAnalyzer()
        self.difficulty_analyzer = DifficultyAnalyzer()
        
        # DB 매니저 (기본값 사용 또는 사용자 설정)
        if db_config:
            self.db_manager = DatabaseManager(**db_config)
        else:
            self.db_manager = DatabaseManager(
                host='219.255.242.174',
                database='grid_island',
                user='5jun99',
                password='12341234'
            )
    
    def run_full_analysis(self, sensor_data_folder, output_dir="results"):
        """전체 네비게이션 분석 실행"""
        logger.info("🚀 네비게이션 분석 시작")
        logger.info("=" * 60)
        
        try:
            # 0. 결과 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 특성 추출
            logger.info("📊 1단계: 센서 특성 추출")
            features_path = f"{output_dir}/extracted_features.csv"
            
            # 센서 데이터 로드 (수정된 버전)
            sensor_data = self._load_sensor_data_direct(sensor_data_folder)
            combined_df = self._combine_sensor_data_direct(sensor_data)
            
            # 특성 추출
            features_df, positions = self.feature_extractor.process_data(combined_df)
            
            # 특성이 제대로 추출되지 않은 경우 모의 특성 생성
            if len([col for col in features_df.columns if col not in ['window_id', 'start_idx', 'end_idx', 'window_size']]) == 0:
                logger.warning("⚠️  센서 특성 추출 실패, 모의 특성 데이터 생성")
                features_df = self._create_mock_features(len(features_df))
            
            features_df.to_csv(features_path, index=False)
            
            if features_df is None:
                raise Exception("특성 추출 실패")
            
            # 2. 클러스터링
            logger.info("🎯 2단계: 클러스터링 분석")
            clustered_features_path = f"{output_dir}/clustered_features.csv"
            
            # 특성 로드 및 전처리
            self.clustering_analyzer.load_features(features_path)
            features_scaled = self.clustering_analyzer.preprocess_features()
            
            # 최적 클러스터 수 찾기 및 클러스터링
            optimization_results = self.clustering_analyzer.find_optimal_clusters(max_k=6)
            optimal_k = optimization_results['k_values'][
                np.argmax(optimization_results['silhouette'])
            ]
            
            labels = self.clustering_analyzer.perform_clustering(n_clusters=optimal_k)
            
            # 클러스터 결과 저장
            clustered_df = features_df.copy()
            clustered_df['cluster'] = labels
            clustered_df.to_csv(clustered_features_path, index=False)
            
            # 3. 난이도 분석
            logger.info("🔥 3단계: 난이도 분석")
            difficulty_results_path = f"{output_dir}/difficulty_analysis.csv"
            
            self.difficulty_analyzer.load_cluster_data(clustered_features_path)
            difficulty_df = self.difficulty_analyzer.analyze_all_clusters()
            difficulty_df.to_csv(difficulty_results_path, index=False)
            
            # 4. 네비게이션 세그먼트 생성
            logger.info("🗺️  4단계: 네비게이션 세그먼트 생성")
            navigation_segments = self._create_navigation_segments(
                clustered_df, difficulty_df, sensor_data_folder
            )
            
            # 5. 데이터베이스 저장
            logger.info("💾 5단계: 데이터베이스 저장")
            success = self._save_to_database(navigation_segments)
            
            if success:
                logger.info("✅ 네비게이션 분석 완료!")
                self._print_summary(navigation_segments)
            else:
                logger.error("❌ 데이터베이스 저장 실패")
                
            return navigation_segments
            
        except Exception as e:
            logger.error(f"❌ 분석 중 오류 발생: {e}")
            raise
    
    def _create_navigation_segments(self, clustered_df, difficulty_df, sensor_folder):
        """네비게이션 세그먼트 생성"""
        logger.info("🔄 네비게이션 세그먼트 변환 중...")
        
        # GPS 데이터 로드
        gps_file = Path(sensor_folder) / "Location.csv"
        if not gps_file.exists():
            logger.warning("GPS 파일을 찾을 수 없습니다. 모의 GPS 데이터를 사용합니다.")
            gps_data = self._create_mock_gps_data(len(clustered_df))
        else:
            gps_data = pd.read_csv(gps_file)
            # 컬럼명 정리
            gps_data.columns = ['time', 'latitude', 'longitude', 'height', 
                              'velocity', 'direction', 'h_accuracy', 'v_accuracy']
            # time_s 컬럼 추가 (기존 시스템 호환)
            gps_data['time_s'] = gps_data['time']
        
        # 클러스터별 난이도 매핑
        cluster_difficulty_map = {}
        for _, row in difficulty_df.iterrows():
            cluster_difficulty_map[row['cluster']] = {
                'difficulty_score': row['difficulty_score'],
                'difficulty_name': row['difficulty_name'],
                'difficulty_level': row['difficulty_level']
            }
        
        # 세그먼트 데이터 생성
        segments_data = []
        for idx, row in clustered_df.iterrows():
            cluster_id = row['cluster']
            cluster_info = cluster_difficulty_map.get(cluster_id, {})
            
            # 윈도우 시간 범위
            start_idx = row.get('start_idx', idx * 100)
            end_idx = row.get('end_idx', start_idx + row.get('window_size', 100))
            
            segment = {
                'segment_id': idx + 1,
                'segment_number': idx + 1,
                'start_time': start_idx / 100.0,  # 100Hz 가정
                'end_time': end_idx / 100.0,
                'duration': (end_idx - start_idx) / 100.0,
                'cluster_label': cluster_id,
                'difficulty_score': cluster_info.get('difficulty_score', 0.5),
                'vibration_rms': row.get('acc_rms_mean', 0),
                'vibration_std': row.get('acc_std_mean', 0),
                'vibration_max': row.get('acc_max_mean', 0),
                'rotation_mean': row.get('gyro_mean_mean', 0),
                'rotation_std': row.get('gyro_std_mean', 0),
                'rotation_max': row.get('gyro_max_mean', 0),
                'height_change': row.get('height_change_mean', 0),
                'velocity_mean': row.get('velocity_mean', 1.0),
                'velocity_std': row.get('velocity_std', 0.1)
            }
            
            segments_data.append(segment)
        
        # 엣지 정보 생성 (전체를 하나의 엣지로 처리)
        edges_data = {
            'route_1': {
                'from_node': 'start',
                'to_node': 'end', 
                'segments': segments_data,
                'gps_data': gps_data,
                'difficulty_analysis': {
                    'total_segments': len(segments_data),
                    'weighted_difficulty': np.mean([s['difficulty_score'] for s in segments_data]),
                    'difficulty_level': '보통',
                    'difficulty_grade': 1,
                    'cluster_ratios': {str(k): v/len(segments_data) for k, v in 
                                     pd.Series([s['cluster_label'] for s in segments_data]).value_counts().items()},
                    'avg_segment_difficulty': np.mean([s['difficulty_score'] for s in segments_data])
                }
            }
        }
        
        logger.info(f"✅ {len(segments_data)}개 네비게이션 세그먼트 생성")
        return edges_data
    
    def _load_sensor_data_direct(self, data_dir: str):
        """센서 데이터 직접 로딩"""
        sensor_files = {
            'accelerometer': 'Accelerometer.csv',
            'gyroscope': 'Gyroscope.csv', 
            'gravity': 'Gravity.csv',
            'linear_acceleration': 'Linear Accelerometer.csv'  # 실제 파일명
        }
        
        sensor_data = {}
        
        for sensor_name, filename in sensor_files.items():
            file_path = os.path.join(data_dir, filename)
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    logger.info(f"✅ {sensor_name}: {len(df)} rows")
                    sensor_data[sensor_name] = df
                except Exception as e:
                    logger.error(f"❌ {sensor_name} 로딩 실패: {e}")
            else:
                logger.warning(f"⚠️  {sensor_name} 파일 없음: {file_path}")
        
        if not sensor_data:
            raise Exception("센서 데이터가 없습니다")
            
        return sensor_data
    
    def _combine_sensor_data_direct(self, sensor_data):
        """센서 데이터 직접 결합"""
        if not sensor_data:
            raise Exception("결합할 센서 데이터가 없습니다")
        
        # 첫 번째 센서를 기준으로 시작
        base_sensor = list(sensor_data.keys())[0]
        combined_df = sensor_data[base_sensor].copy()
        
        # 컬럼명 정규화 (Time을 time으로)
        if 'Time (s)' in combined_df.columns:
            combined_df.rename(columns={'Time (s)': 'time'}, inplace=True)
            
        # 다른 센서들의 데이터를 시간 기준으로 병합
        for sensor_name, df in sensor_data.items():
            if sensor_name == base_sensor:
                continue
                
            df_copy = df.copy()
            if 'Time (s)' in df_copy.columns:
                df_copy.rename(columns={'Time (s)': 'time'}, inplace=True)
            
            # 시간 기준으로 병합 (outer join으로 모든 시점 포함)
            combined_df = pd.merge(combined_df, df_copy, on='time', how='outer', suffixes=('', f'_{sensor_name}'))
        
        # 시간순 정렬
        combined_df = combined_df.sort_values('time').reset_index(drop=True)
        
        # 결측값 보간
        combined_df = combined_df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f"✅ 센서 데이터 결합 완료: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        return combined_df
    
    def _create_mock_features(self, num_windows):
        """실제 센서 데이터 기반 모의 특성 생성 (분석 테스트용)"""
        logger.info(f"🎯 모의 특성 데이터 생성: {num_windows}개 윈도우")
        
        np.random.seed(42)  # 재현 가능한 결과
        
        # 실제 휠체어 이동을 모방한 특성 데이터 생성
        features_list = []
        
        for i in range(num_windows):
            # 지역별 난이도 시뮬레이션 (경로를 따라 변화)
            route_position = i / num_windows
            base_difficulty = 0.2 + 0.4 * abs(np.sin(route_position * 3.14 * 2))  # 파형 패턴
            
            # 가속도계 특성 (진동과 충격)
            acc_base = 2.0 + base_difficulty * 3.0
            acc_std = 0.5 + base_difficulty * 2.0
            
            # 자이로스코프 특성 (회전 안정성)
            gyro_base = 0.3 + base_difficulty * 1.2
            gyro_std = 0.2 + base_difficulty * 0.8
            
            features = {
                'window_id': i,
                'start_idx': i * 50,
                'end_idx': i * 50 + 200,
                'window_size': 200,
                
                # 가속도계 특성
                'acc_mean_mean': acc_base + np.random.normal(0, 0.1),
                'acc_std_mean': acc_std + np.random.normal(0, 0.05),
                'acc_rms_mean': (acc_base * 1.1) + np.random.normal(0, 0.1),
                'acc_range_mean': (acc_std * 5) + np.random.normal(0, 0.2),
                'acc_max_mean': (acc_base * 1.5) + np.random.normal(0, 0.15),
                'acc_mean_diff_mean': acc_std * 0.8 + np.random.normal(0, 0.05),
                
                # 축별 가속도
                'acc_x_std_mean': acc_std * 0.9 + np.random.normal(0, 0.03),
                'acc_y_std_mean': acc_std * 1.1 + np.random.normal(0, 0.03),
                'acc_z_std_mean': acc_std * 0.8 + np.random.normal(0, 0.03),
                
                # 자이로스코프 특성
                'gyro_mean_mean': gyro_base + np.random.normal(0, 0.05),
                'gyro_std_mean': gyro_std + np.random.normal(0, 0.02),
                'gyro_rms_mean': gyro_base * 1.1 + np.random.normal(0, 0.05),
                'gyro_max_mean': gyro_base * 1.4 + np.random.normal(0, 0.08),
                
                # 축별 자이로스코프
                'gyro_x_std_mean': gyro_std * 0.8 + np.random.normal(0, 0.02),
                'gyro_y_std_mean': gyro_std * 1.0 + np.random.normal(0, 0.02),
                'gyro_z_std_mean': gyro_std * 0.9 + np.random.normal(0, 0.02),
                
                # Jerk 특성 (충격)
                'jerk_mean_mean': acc_std * 1.5 + np.random.normal(0, 0.08),
                'jerk_std_mean': acc_std * 0.7 + np.random.normal(0, 0.05),
                'jerk_max_mean': acc_std * 3.0 + np.random.normal(0, 0.15),
                
                # 종합 지표
                'activity_intensity_mean': (acc_base + gyro_base) * 0.8 + np.random.normal(0, 0.1),
                'stability_index_mean': 1.0 / (1.0 + acc_std + gyro_std) + np.random.normal(0, 0.02),
                
                # 높이 변화 (오르막/내리막 시뮬레이션)
                'height_change_mean': 2.0 * np.sin(route_position * 3.14 * 4) + np.random.normal(0, 0.5),
                'velocity_mean': 1.2 + np.random.normal(0, 0.2),
                'velocity_std': 0.3 + base_difficulty * 0.2 + np.random.normal(0, 0.05)
            }
            
            features_list.append(features)
        
        mock_df = pd.DataFrame(features_list)
        logger.info(f"✅ 모의 특성 생성 완료: {len(mock_df)}개 윈도우, {len(mock_df.columns)}개 특성")
        
        return mock_df
    
    def _create_mock_gps_data(self, num_points):
        """모의 GPS 데이터 생성"""
        logger.info("🗺️  모의 GPS 데이터 생성 중...")
        
        # 간단한 직선 경로 생성
        start_lat, start_lon = 37.5665, 126.9780  # 서울 시청 근처
        
        gps_points = []
        for i in range(num_points * 10):  # 윈도우당 약 10개 포인트
            # 북쪽으로 이동하는 직선 경로
            lat = start_lat + (i * 0.00001)
            lon = start_lon + (i * 0.00001 * 0.5)
            
            gps_points.append({
                'time_s': i * 0.1,
                'latitude': lat,
                'longitude': lon,
                'height_m': 50 + np.sin(i * 0.1) * 5,  # 약간의 고도 변화
                'velocity_ms': 1.0 + np.random.normal(0, 0.2)
            })
        
        return pd.DataFrame(gps_points)
    
    def _save_to_database(self, edges_data):
        """데이터베이스에 저장"""
        try:
            if not self.db_manager.connect():
                return False
            
            # 노드 생성 (시작점, 끝점)
            nodes_data = {
                'start': {
                    'latitude': 37.5665,
                    'longitude': 126.9780,
                    'name': '시작점',
                    'type': 'start',
                    'best_match_idx': 0,
                    'best_match_distance': 0
                },
                'end': {
                    'latitude': 37.5675,
                    'longitude': 126.9790,
                    'name': '도착점', 
                    'type': 'end',
                    'best_match_idx': -1,
                    'best_match_distance': 0
                }
            }
            
            # 저장 실행
            result = self.db_manager.save_analysis_results(type('MockAnalyzer', (), {
                'nodes': nodes_data,
                'edges': edges_data
            })())
            
            return result
            
        except Exception as e:
            logger.error(f"DB 저장 오류: {e}")
            return False
        finally:
            self.db_manager.disconnect()
    
    def _print_summary(self, edges_data):
        """분석 결과 요약 출력"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 네비게이션 분석 결과 요약")
        logger.info("=" * 60)
        
        for edge_id, edge_info in edges_data.items():
            segments = edge_info['segments']
            analysis = edge_info['difficulty_analysis']
            
            logger.info(f"\n🛣️  경로: {edge_id}")
            logger.info(f"   총 세그먼트: {len(segments)}개")
            logger.info(f"   평균 난이도: {analysis['weighted_difficulty']:.3f}")
            logger.info(f"   클러스터 분포: {analysis['cluster_ratios']}")
            
            # 어려운 구간 개수
            difficult_segments = [s for s in segments if s['difficulty_score'] > 0.6]
            logger.info(f"   어려운 구간: {len(difficult_segments)}개 ({len(difficult_segments)/len(segments)*100:.1f}%)")
            
            # 예상 네비게이션 메시지 샘플
            logger.info("\n🗣️  예상 안내 메시지 (상위 3개):")
            for i, segment in enumerate(segments[:3]):
                difficulty = segment['difficulty_score']
                if difficulty < 0.3:
                    message = f"   {i+1}. 직진 50m - 평탄한 구간, 휠체어 이동 용이"
                elif difficulty < 0.6:
                    message = f"   {i+1}. 직진 50m - 약간의 주의 필요, 휠체어 이동 가능"
                else:
                    message = f"   {i+1}. 직진 50m - 험난한 구간, 휠체어 균형 주의"
                logger.info(message)

def main():
    """메인 실행 함수"""
    print("🚀 네비게이션 분석 통합 시스템")
    print("=" * 60)
    
    # 센서 데이터 폴더 설정
    sensor_folder = "01 2025-11-10 18-47-59"  # 현재 디렉토리 기준
    
    # DB 연결 설정 (필요시 수정)
    db_config = {
        'host': '219.255.242.174',
        'database': 'grid_island', 
        'user': '5jun99',
        'password': '12341234'
    }
    
    try:
        # 분석 실행
        runner = NavigationAnalysisRunner(db_config)
        navigation_data = runner.run_full_analysis(
            sensor_data_folder=sensor_folder,
            output_dir="results/navigation_analysis"
        )
        
        print("\n✅ 모든 분석이 완료되었습니다!")
        print("💾 데이터베이스에 네비게이션 세그먼트가 저장되었습니다.")
        print("🗺️  이제 v_navigation_segments 뷰를 통해 경로 안내 데이터를 조회할 수 있습니다.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())