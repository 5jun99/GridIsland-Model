#!/usr/bin/env python3
"""
간선별 데이터 분석 및 DB 저장 도구
간선 1,2,3 데이터를 분석하여 데이터베이스에 저장합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
import os
import sys
import math
from pathlib import Path
from database_manager import DatabaseManager
from nearest_node_finder import NearestNodeFinder
import logging

# 새로운 분석 시스템 모듈 import 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "new_analysis_system"))

try:
    from feature_extractor import FeatureExtractor
    from clustering_analyzer import ClusteringAnalyzer
    from difficulty_analyzer import DifficultyAnalyzer
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"고급 분석 모듈 로드 실패: {e}. 기본 분석만 사용합니다.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeDataSaver:
    """간선별 데이터 분석 및 DB 저장 클래스"""
    
    def __init__(self, node_coord_file: str = "node_coord.txt"):
        """
        Args:
            node_coord_file: 노드 좌표 파일 경로
        """
        self.node_finder = NearestNodeFinder(node_coord_file)
        self.db_manager = DatabaseManager(
            host='219.255.242.174',
            database='grid_island',
            user='5jun99',
            password='12341234'
        )
        
        # 고급 분석기 초기화 (가능한 경우)
        try:
            self.feature_extractor = FeatureExtractor()
            self.clustering_analyzer = ClusteringAnalyzer()
            self.difficulty_analyzer = DifficultyAnalyzer()
            self.use_advanced_analysis = True
            logger.info("🔬 고급 분석 시스템 활성화")
        except:
            self.use_advanced_analysis = False
            logger.info("📊 기본 분석 시스템 사용")
        
        # 간선 매핑 정보 (18개 노드 기준, 1-based 인덱스)
        self.edge_mappings = {
            '01': {'from_node': 8, 'to_node': 9, 'edge_id': 'edge_8_to_9_session01'},
            '02': {'from_node': 9, 'to_node': 8, 'edge_id': 'edge_9_to_8_session02'},
            '03': {'from_node': 8, 'to_node': 10, 'edge_id': 'edge_8_to_10_session03'},
            '04': {'from_node': 10, 'to_node': 8, 'edge_id': 'edge_10_to_8_session04'},
            '05': {'from_node': 7, 'to_node': 8, 'edge_id': 'edge_7_to_8_session05'},
            '06': {'from_node': 8, 'to_node': 7, 'edge_id': 'edge_8_to_7_session06'},
            '07': {'from_node': 6, 'to_node': 7, 'edge_id': 'edge_6_to_7_session07'},
            '08': {'from_node': 7, 'to_node': 6, 'edge_id': 'edge_7_to_6_session08'},
            '09': {'from_node': 5, 'to_node': 18, 'edge_id': 'edge_5_to_18_session09'},
            '10': {'from_node': 18, 'to_node': 5, 'edge_id': 'edge_18_to_5_session10'},
            '11': {'from_node': 4, 'to_node': 3, 'edge_id': 'edge_4_to_3_session11'},
            '12': {'from_node': 3, 'to_node': 4, 'edge_id': 'edge_3_to_4_session12'},
            '13': {'from_node': 11, 'to_node': 3, 'edge_id': 'edge_11_to_3_session13'},
            '14': {'from_node': 3, 'to_node': 11, 'edge_id': 'edge_3_to_11_session14'},
            '15': {'from_node': 12, 'to_node': 3, 'edge_id': 'edge_12_to_3_session15'},
            '16': {'from_node': 3, 'to_node': 12, 'edge_id': 'edge_3_to_12_session16'},
            '19': {'from_node': 3, 'to_node': 13, 'edge_id': 'edge_3_to_13_session19'},
            '20': {'from_node': 13, 'to_node': 3, 'edge_id': 'edge_13_to_3_session20'},
            '23': {'from_node': 11, 'to_node': 2, 'edge_id': 'edge_11_to_2_session23'},
            '24': {'from_node': 2, 'to_node': 11, 'edge_id': 'edge_2_to_11_session24'},
            '25': {'from_node': 1, 'to_node': 2, 'edge_id': 'edge_1_to_2_session25'},
            '26': {'from_node': 2, 'to_node': 1, 'edge_id': 'edge_2_to_1_session26'},
            '29': {'from_node': 15, 'to_node': 1, 'edge_id': 'edge_15_to_1_session29'},
            '30': {'from_node': 1, 'to_node': 15, 'edge_id': 'edge_1_to_15_session30'},
            '31': {'from_node': 18, 'to_node': 13, 'edge_id': 'edge_18_to_13_session31'},
            '32': {'from_node': 13, 'to_node': 18, 'edge_id': 'edge_13_to_18_session32'},
            '39': {'from_node': 16, 'to_node': 1, 'edge_id': 'edge_16_to_1_session39'},
            '40': {'from_node': 1, 'to_node': 16, 'edge_id': 'edge_1_to_16_session40'}
        }
        
        self.nodes_data = {}
        self.edges_data = {}
        
    def load_nodes(self) -> Dict[str, Dict]:
        """노드 좌표 데이터 로드 (1-based 인덱스)"""
        logger.info("📍 노드 데이터 로드 중...")
        
        for i, (lat, lng) in enumerate(self.node_finder.node_coords):
            node_id = f"node_{i + 1}"  # 1-based 인덱스
            self.nodes_data[node_id] = {
                'latitude': lat,
                'longitude': lng,
                'name': f'노드 {i + 1}',
                'type': 'waypoint'
            }
        
        logger.info(f"✅ 노드 로드 완료: {len(self.nodes_data)}개 (노드 1~{len(self.nodes_data)})")
        return self.nodes_data
    
    def analyze_edge_data(self, data_folder: str) -> Dict[str, Any]:
        """개별 간선 데이터 분석"""
        logger.info(f"🔍 간선 데이터 분석: {data_folder}")
        
        # GPS 데이터 로드
        gps_file = Path(data_folder) / "Location.csv"
        
        if not gps_file.exists():
            logger.error(f"❌ GPS 파일 없음: {gps_file}")
            return None
        
        # 파일 구분자 확인 (세미콜론 추가)
        with open(gps_file, 'r') as f:
            first_line = f.readline()
            if ';' in first_line:
                separator = ';'
            elif '\t' in first_line:
                separator = '\t'
            else:
                separator = ','
        
        gps_data = pd.read_csv(gps_file, sep=separator)
        
        # 컬럼명 정리
        gps_data.rename(columns={
            'Time (s)': 'time_s',
            'Latitude (°)': 'latitude',
            'Longitude (°)': 'longitude', 
            'Height (m)': 'height_m',
            'Velocity (m/s)': 'velocity_ms'
        }, inplace=True)
        
        # 센서 데이터 로드 시도
        sensor_data = self._load_sensor_data(data_folder)
        
        # 간선 정보 추출
        data_number = data_folder.split()[0]  # '01', '02' 등
        edge_info = self.edge_mappings.get(data_number)
        
        if not edge_info:
            logger.warning(f"⚠️  매핑되지 않은 데이터: {data_number}")
            return None
        
        # GPS 분석
        gps_analysis = self._analyze_gps_data(gps_data)
        
        # 고급 분석 사용 시 클러스터링 기반 세그먼트 생성
        if self.use_advanced_analysis:
            segments = self._create_clustering_segments(gps_data, sensor_data, data_folder)
        else:
            # 기본 시간 기반 세그먼트 분석
            segments = self._create_time_segments(gps_data, sensor_data)
        
        # 난이도 분석
        difficulty_analysis = self._analyze_difficulty(segments, gps_data)
        
        # 네비게이션 세그먼트 생성 (새로운 기능)
        navigation_segments = self._create_navigation_segments(segments, gps_data)
        
        edge_data = {
            'edge_id': edge_info['edge_id'],
            'from_node': f"node_{edge_info['from_node']}",
            'to_node': f"node_{edge_info['to_node']}",
            'data_folder': data_folder,
            'gps_data': gps_data,
            'sensor_data': sensor_data,
            'gps_analysis': gps_analysis,
            'segments': segments,
            'difficulty_analysis': difficulty_analysis,
            'navigation_segments': navigation_segments  # 네비게이션 세그먼트 추가
        }
        
        logger.info(f"✅ 간선 분석 완료: {edge_info['edge_id']}")
        return edge_data
    
    def _load_sensor_data(self, data_folder: str) -> Dict[str, pd.DataFrame]:
        """센서 데이터 로드"""
        sensor_files = ['Accelerometer.csv', 'Gyroscope.csv']
        sensor_data = {}
        
        for sensor_file in sensor_files:
            file_path = Path(data_folder) / sensor_file
            if file_path.exists():
                try:
                    # 구분자 확인 (세미콜론 추가)
                    with open(file_path, 'r') as f:
                        first_line = f.readline()
                        if ';' in first_line:
                            separator = ';'
                        elif '\t' in first_line:
                            separator = '\t'
                        else:
                            separator = ','
                    
                    df = pd.read_csv(file_path, sep=separator)
                    sensor_data[sensor_file.replace('.csv', '')] = df
                    logger.info(f"   📱 {sensor_file} 로드: {len(df)}개 샘플")
                except Exception as e:
                    logger.warning(f"⚠️  {sensor_file} 로드 실패: {e}")
        
        return sensor_data
    
    def _analyze_gps_data(self, gps_data: pd.DataFrame) -> Dict[str, Any]:
        """GPS 데이터 분석"""
        analysis = {
            'total_points': len(gps_data),
            'duration': gps_data['time_s'].max() - gps_data['time_s'].min(),
            'start_time': gps_data['time_s'].min(),
            'end_time': gps_data['time_s'].max(),
            'lat_range': [gps_data['latitude'].min(), gps_data['latitude'].max()],
            'lng_range': [gps_data['longitude'].min(), gps_data['longitude'].max()],
        }
        
        # 거리 계산
        if len(gps_data) > 1:
            start_lat, start_lng = gps_data.iloc[0]['latitude'], gps_data.iloc[0]['longitude']
            end_lat, end_lng = gps_data.iloc[-1]['latitude'], gps_data.iloc[-1]['longitude']
            analysis['path_distance'] = self._calculate_distance(start_lat, start_lng, end_lat, end_lng)
        else:
            analysis['path_distance'] = 0
        
        # 속도 통계 (NaN 제외)
        if 'velocity_ms' in gps_data.columns:
            velocity_clean = gps_data['velocity_ms'].dropna()
            if len(velocity_clean) > 0:
                analysis['avg_velocity'] = velocity_clean.mean()
                analysis['max_velocity'] = velocity_clean.max()
            else:
                analysis['avg_velocity'] = 0
                analysis['max_velocity'] = 0
        
        return analysis
    
    def _create_time_segments(self, gps_data: pd.DataFrame, sensor_data: Dict) -> List[Dict]:
        """시간 기반 세그먼트 생성"""
        segments = []
        
        # 간단한 시간 기반 세그먼트 (10초 단위)
        segment_duration = 10.0  # 10초
        start_time = gps_data['time_s'].min()
        end_time = gps_data['time_s'].max()
        
        segment_id = 1
        current_time = start_time
        
        while current_time < end_time:
            segment_end = min(current_time + segment_duration, end_time)
            
            # 해당 시간 구간의 GPS 데이터
            segment_gps = gps_data[
                (gps_data['time_s'] >= current_time) & 
                (gps_data['time_s'] < segment_end)
            ]
            
            if len(segment_gps) == 0:
                current_time = segment_end
                continue
            
            # 세그먼트 분석
            segment_analysis = self._analyze_segment(segment_gps, sensor_data, current_time, segment_end)
            segment_analysis['segment_id'] = segment_id
            segment_analysis['start_time'] = current_time
            segment_analysis['end_time'] = segment_end
            
            segments.append(segment_analysis)
            
            current_time = segment_end
            segment_id += 1
        
        logger.info(f"   📊 세그먼트 생성: {len(segments)}개")
        return segments
    
    def _analyze_segment(self, gps_data: pd.DataFrame, sensor_data: Dict, start_time: float, end_time: float) -> Dict:
        """개별 세그먼트 분석"""
        analysis = {
            'gps_points': len(gps_data),
            'duration': end_time - start_time
        }
        
        # GPS 기반 분석
        if len(gps_data) > 0:
            if 'velocity_ms' in gps_data.columns:
                velocity_clean = gps_data['velocity_ms'].dropna()
                analysis['velocity_mean'] = velocity_clean.mean() if len(velocity_clean) > 0 else 0
                analysis['velocity_std'] = velocity_clean.std() if len(velocity_clean) > 0 else 0
            else:
                analysis['velocity_mean'] = 0
                analysis['velocity_std'] = 0
            
            if 'height_m' in gps_data.columns:
                height_clean = gps_data['height_m'].dropna()
                if len(height_clean) > 1:
                    analysis['height_change'] = height_clean.max() - height_clean.min()
                else:
                    analysis['height_change'] = 0
            else:
                analysis['height_change'] = 0
        
        # 센서 데이터 기반 분석 (가속도계/자이로스코프)
        analysis.update(self._analyze_sensor_segment(sensor_data, start_time, end_time))
        
        # 간단한 난이도 점수 계산
        analysis['difficulty_score'] = self._calculate_simple_difficulty(analysis)
        
        return analysis
    
    def _analyze_sensor_segment(self, sensor_data: Dict, start_time: float, end_time: float) -> Dict:
        """센서 데이터 세그먼트 분석"""
        analysis = {
            'vibration_rms': 0.0,
            'vibration_std': 0.0, 
            'vibration_max': 0.0,
            'rotation_mean': 0.0,
            'rotation_std': 0.0,
            'rotation_max': 0.0
        }
        
        # 가속도계 분석
        if 'Accelerometer' in sensor_data:
            acc_data = sensor_data['Accelerometer']
            
            # 컬럼명 확인 및 정리
            acc_cols = acc_data.columns.tolist()
            if len(acc_cols) >= 4:  # time + x,y,z
                time_col = acc_cols[0]
                x_col, y_col, z_col = acc_cols[1], acc_cols[2], acc_cols[3]
                
                # 시간 구간 필터링
                segment_acc = acc_data[
                    (acc_data[time_col] >= start_time) & 
                    (acc_data[time_col] < end_time)
                ]
                
                if len(segment_acc) > 0:
                    # 진동 분석 (가속도 크기)
                    acc_magnitude = np.sqrt(
                        segment_acc[x_col]**2 + 
                        segment_acc[y_col]**2 + 
                        segment_acc[z_col]**2
                    )
                    
                    analysis['vibration_rms'] = np.sqrt(np.mean(acc_magnitude**2))
                    analysis['vibration_std'] = acc_magnitude.std()
                    analysis['vibration_max'] = acc_magnitude.max()
        
        # 자이로스코프 분석
        if 'Gyroscope' in sensor_data:
            gyro_data = sensor_data['Gyroscope']
            
            gyro_cols = gyro_data.columns.tolist()
            if len(gyro_cols) >= 4:
                time_col = gyro_cols[0]
                x_col, y_col, z_col = gyro_cols[1], gyro_cols[2], gyro_cols[3]
                
                segment_gyro = gyro_data[
                    (gyro_data[time_col] >= start_time) & 
                    (gyro_data[time_col] < end_time)
                ]
                
                if len(segment_gyro) > 0:
                    # 회전 분석
                    rotation_magnitude = np.sqrt(
                        segment_gyro[x_col]**2 + 
                        segment_gyro[y_col]**2 + 
                        segment_gyro[z_col]**2
                    )
                    
                    analysis['rotation_mean'] = rotation_magnitude.mean()
                    analysis['rotation_std'] = rotation_magnitude.std()
                    analysis['rotation_max'] = rotation_magnitude.max()
        
        return analysis
    
    def _calculate_simple_difficulty(self, segment_analysis: Dict) -> float:
        """간단한 난이도 점수 계산 (0-1 범위)"""
        # 가중치 기반 난이도 계산
        weights = {
            'vibration': 0.3,
            'rotation': 0.3,
            'velocity_variation': 0.2,
            'height_change': 0.2
        }
        
        # 각 요소를 0-1로 정규화
        vibration_score = min(1.0, segment_analysis.get('vibration_rms', 0) / 20.0)  # 20m/s² 기준
        rotation_score = min(1.0, segment_analysis.get('rotation_mean', 0) / 5.0)    # 5rad/s 기준
        velocity_var_score = min(1.0, segment_analysis.get('velocity_std', 0) / 3.0) # 3m/s 기준
        height_score = min(1.0, segment_analysis.get('height_change', 0) / 10.0)    # 10m 기준
        
        difficulty_score = (
            weights['vibration'] * vibration_score +
            weights['rotation'] * rotation_score +
            weights['velocity_variation'] * velocity_var_score +
            weights['height_change'] * height_score
        )
        
        return min(1.0, difficulty_score)
    
    def _create_clustering_segments(self, gps_data: pd.DataFrame, sensor_data: Dict, data_folder: str) -> List[Dict]:
        """클러스터링 기반 세그먼트 생성 (고급 분석)"""
        logger.info("   🔬 클러스터링 기반 세그먼트 분석")
        
        try:
            # 센서 데이터 결합
            combined_df = self._combine_sensor_data_for_clustering(sensor_data)
            
            if combined_df is None or len(combined_df) == 0:
                logger.warning("   ⚠️  센서 데이터 결합 실패, 기본 분석으로 전환")
                return self._create_time_segments(gps_data, sensor_data)
            
            # 특성 추출
            features_df, positions = self.feature_extractor.process_data(combined_df)
            
            # 특성이 부족한 경우 mock 특성 생성
            if len([col for col in features_df.columns if col not in ['window_id', 'start_idx', 'end_idx', 'window_size']]) == 0:
                features_df = self._create_mock_features_for_segments(len(features_df))
            
            # 클러스터링 - 특성 파일로 저장 후 로드
            temp_features_path = "/tmp/temp_features.csv"
            features_df.to_csv(temp_features_path, index=False)
            self.clustering_analyzer.load_features(temp_features_path)
            features_scaled = self.clustering_analyzer.preprocess_features()
            
            # 최적 클러스터 수 찾기
            optimization_results = self.clustering_analyzer.find_optimal_clusters(max_k=6)
            optimal_k = optimization_results['k_values'][
                np.argmax(optimization_results['silhouette'])
            ]
            
            labels = self.clustering_analyzer.perform_clustering(n_clusters=optimal_k)
            
            # 클러스터 결과를 특성에 추가
            features_df['cluster'] = labels
            
            # 난이도 분석 - 클러스터 데이터로 저장 후 로드
            temp_clustered_path = "/tmp/temp_clustered.csv"
            features_df.to_csv(temp_clustered_path, index=False)
            self.difficulty_analyzer.load_cluster_data(temp_clustered_path)
            difficulty_df = self.difficulty_analyzer.analyze_all_clusters()
            
            # 클러스터별 난이도 매핑
            cluster_difficulty_map = {}
            for _, row in difficulty_df.iterrows():
                cluster_difficulty_map[row['cluster']] = {
                    'difficulty_score': row['difficulty_score'],
                    'difficulty_name': row['difficulty_name'],
                    'difficulty_level': row['difficulty_level']
                }
            
            # 세그먼트 생성
            segments = []
            for idx, row in features_df.iterrows():
                cluster_id = row['cluster']
                cluster_info = cluster_difficulty_map.get(cluster_id, {})
                
                start_idx = row.get('start_idx', idx * 100)
                end_idx = row.get('end_idx', start_idx + row.get('window_size', 100))
                
                segment = {
                    'segment_id': idx + 1,
                    'start_time': start_idx / 50.0,  # 50Hz 가정
                    'end_time': end_idx / 50.0,
                    'duration': (end_idx - start_idx) / 50.0,
                    'cluster_label': cluster_id,
                    'difficulty_score': cluster_info.get('difficulty_score', 0.5),
                    'vibration_rms': row.get('acc_rms', 0),
                    'vibration_std': row.get('acc_std', 0),
                    'vibration_max': row.get('acc_max', 0),
                    'rotation_mean': row.get('gyro_mean', 0),
                    'rotation_std': row.get('gyro_std', 0),
                    'rotation_max': row.get('gyro_max', 0),
                    'height_change': 0,  # GPS에서 계산
                    'velocity_mean': 1.0,
                    'velocity_std': 0.1,
                    'gps_points': 0
                }
                
                segments.append(segment)
            
            logger.info(f"   ✅ 클러스터링 세그먼트 생성: {len(segments)}개 ({optimal_k}개 클러스터)")
            return segments
            
        except Exception as e:
            logger.error(f"   ❌ 클러스터링 분석 실패: {e}")
            return self._create_time_segments(gps_data, sensor_data)
    
    def _combine_sensor_data_for_clustering(self, sensor_data: Dict) -> pd.DataFrame:
        """클러스터링을 위한 센서 데이터 결합"""
        if not sensor_data:
            return None
        
        # 첫 번째 센서를 기준으로 시작
        base_sensor = list(sensor_data.keys())[0]
        combined_df = sensor_data[base_sensor].copy()
        
        # 컬럼명 정규화
        time_cols = ['Time (s)', 'time', 'Time']
        time_col = None
        for col in time_cols:
            if col in combined_df.columns:
                time_col = col
                break
        
        if time_col and time_col != 'time':
            combined_df.rename(columns={time_col: 'time'}, inplace=True)
        
        # 센서별 컬럼명 정리
        if 'Accelerometer' in sensor_data:
            acc_df = sensor_data['Accelerometer'].copy()
            if time_col and time_col in acc_df.columns:
                acc_df.rename(columns={time_col: 'time'}, inplace=True)
            # 가속도 컬럼 이름 정규화
            acc_cols = [col for col in acc_df.columns if col != 'time']
            if len(acc_cols) >= 3:
                acc_df.rename(columns={
                    acc_cols[0]: 'acc_x',
                    acc_cols[1]: 'acc_y', 
                    acc_cols[2]: 'acc_z'
                }, inplace=True)
        
        if 'Gyroscope' in sensor_data:
            gyro_df = sensor_data['Gyroscope'].copy()
            if time_col and time_col in gyro_df.columns:
                gyro_df.rename(columns={time_col: 'time'}, inplace=True)
            # 자이로 컬럼 이름 정규화
            gyro_cols = [col for col in gyro_df.columns if col != 'time']
            if len(gyro_cols) >= 3:
                gyro_df.rename(columns={
                    gyro_cols[0]: 'gyro_x',
                    gyro_cols[1]: 'gyro_y',
                    gyro_cols[2]: 'gyro_z'
                }, inplace=True)
        
        # 데이터 병합
        if 'Accelerometer' in sensor_data and 'Gyroscope' in sensor_data:
            combined_df = pd.merge(sensor_data['Accelerometer'], sensor_data['Gyroscope'], on='time', how='outer')
        elif 'Accelerometer' in sensor_data:
            combined_df = sensor_data['Accelerometer']
        elif 'Gyroscope' in sensor_data:
            combined_df = sensor_data['Gyroscope']
        
        # 결측값 처리
        combined_df = combined_df.sort_values('time').interpolate().fillna(method='bfill').fillna(method='ffill')
        
        return combined_df
    
    def _create_mock_features_for_segments(self, num_windows: int) -> pd.DataFrame:
        """세그먼트용 모의 특성 생성"""
        np.random.seed(42)
        
        features_list = []
        for i in range(num_windows):
            # 경로상 위치에 따른 난이도 변화
            route_position = i / num_windows
            base_difficulty = 0.2 + 0.4 * abs(np.sin(route_position * 3.14 * 2))
            
            features = {
                'window_id': i,
                'start_idx': i * 50,
                'end_idx': i * 50 + 200,
                'window_size': 200,
                'acc_mean': 2.0 + base_difficulty * 3.0 + np.random.normal(0, 0.1),
                'acc_std': 0.5 + base_difficulty * 2.0 + np.random.normal(0, 0.05),
                'acc_rms': 2.2 + base_difficulty * 3.3 + np.random.normal(0, 0.1),
                'acc_max': 3.0 + base_difficulty * 4.5 + np.random.normal(0, 0.15),
                'gyro_mean': 0.3 + base_difficulty * 1.2 + np.random.normal(0, 0.05),
                'gyro_std': 0.2 + base_difficulty * 0.8 + np.random.normal(0, 0.02),
                'gyro_max': 0.4 + base_difficulty * 1.6 + np.random.normal(0, 0.08),
                'activity_intensity': (2.0 + base_difficulty * 2.0) + np.random.normal(0, 0.1),
                'stability_index': 1.0 / (1.0 + base_difficulty + 0.5) + np.random.normal(0, 0.02)
            }
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _create_navigation_segments(self, segments: List[Dict], gps_data: pd.DataFrame) -> List[Dict]:
        """네비게이션 세그먼트 생성 및 병합"""
        if not segments:
            return []
        
        logger.info("   🗺️  네비게이션 세그먼트 생성 중...")
        
        # GPS 기반 위치 정보 추가
        enriched_segments = self._enrich_segments_with_gps(segments, gps_data)
        
        # 유사한 연속 세그먼트 병합
        merged_segments = self._merge_similar_segments(enriched_segments)
        
        # 네비게이션 지시사항 생성
        navigation_segments = self._generate_navigation_instructions(merged_segments)
        
        logger.info(f"   ✅ 네비게이션 세그먼트: {len(segments)}개 → {len(navigation_segments)}개로 병합")
        return navigation_segments
    
    def _enrich_segments_with_gps(self, segments: List[Dict], gps_data: pd.DataFrame) -> List[Dict]:
        """GPS 정보로 세그먼트 보강"""
        enriched = []
        
        for segment in segments:
            start_time = segment['start_time']
            end_time = segment['end_time']
            
            # 해당 시간 범위의 GPS 데이터
            segment_gps = gps_data[
                (gps_data['time_s'] >= start_time) & 
                (gps_data['time_s'] <= end_time)
            ] if 'time_s' in gps_data.columns else gps_data.iloc[:1]
            
            if len(segment_gps) > 0:
                start_gps = segment_gps.iloc[0]
                end_gps = segment_gps.iloc[-1] if len(segment_gps) > 1 else start_gps
                
                # GPS 정보 추가
                segment.update({
                    'start_lat': start_gps['latitude'] if 'latitude' in start_gps else 37.5665,
                    'start_lon': start_gps['longitude'] if 'longitude' in start_gps else 126.9780,
                    'end_lat': end_gps['latitude'] if 'latitude' in end_gps else 37.5665,
                    'end_lon': end_gps['longitude'] if 'longitude' in end_gps else 126.9780
                })
                
                # 거리 및 방향 계산
                distance = self._calculate_distance(
                    segment['start_lat'], segment['start_lon'],
                    segment['end_lat'], segment['end_lon']
                )
                bearing = self._calculate_bearing(
                    segment['start_lat'], segment['start_lon'],
                    segment['end_lat'], segment['end_lon']
                )
                
                segment.update({
                    'distance_meters': distance,
                    'bearing_degrees': bearing,
                    'estimated_time_sec': segment['duration']
                })
            else:
                # 기본값 설정
                segment.update({
                    'start_lat': 37.5665, 'start_lon': 126.9780,
                    'end_lat': 37.5665, 'end_lon': 126.9780,
                    'distance_meters': 10.0,
                    'bearing_degrees': 0.0,
                    'estimated_time_sec': segment['duration']
                })
            
            enriched.append(segment)
        
        return enriched
    
    def _merge_similar_segments(self, segments: List[Dict]) -> List[Dict]:
        """유사한 연속 세그먼트 병합"""
        if len(segments) <= 1:
            return segments
        
        merged = []
        current_segment = segments[0].copy()
        original_ids = [current_segment.get('segment_id', 1)]
        
        for next_segment in segments[1:]:
            # 병합 조건 확인
            difficulty_diff = abs(current_segment['difficulty_score'] - next_segment['difficulty_score'])
            time_gap = next_segment['start_time'] - current_segment['end_time']
            
            if difficulty_diff < 0.15 and time_gap < 10.0:  # 유사한 난이도, 10초 이내
                # 세그먼트 병합
                current_segment.update({
                    'end_time': next_segment['end_time'],
                    'end_lat': next_segment['end_lat'],
                    'end_lon': next_segment['end_lon'],
                    'duration': current_segment['duration'] + next_segment['duration'],
                    'distance_meters': current_segment['distance_meters'] + next_segment['distance_meters'],
                    'estimated_time_sec': current_segment['estimated_time_sec'] + next_segment['estimated_time_sec']
                })
                original_ids.append(next_segment.get('segment_id', len(original_ids) + 1))
            else:
                # 현재 세그먼트 완료
                current_segment['original_segment_ids'] = original_ids
                current_segment['is_merged'] = len(original_ids) > 1
                merged.append(current_segment)
                
                # 새 세그먼트 시작
                current_segment = next_segment.copy()
                original_ids = [current_segment.get('segment_id', len(merged) + 1)]
        
        # 마지막 세그먼트 추가
        current_segment['original_segment_ids'] = original_ids
        current_segment['is_merged'] = len(original_ids) > 1
        merged.append(current_segment)
        
        return merged
    
    def _generate_navigation_instructions(self, segments: List[Dict]) -> List[Dict]:
        """네비게이션 지시사항 생성"""
        navigation_segments = []
        prev_bearing = None
        
        for i, segment in enumerate(segments):
            # 회전 각도 계산
            turn_angle = 0.0
            if prev_bearing is not None:
                turn_angle = self._calculate_turn_angle(prev_bearing, segment['bearing_degrees'])
            
            # 방향 지시 생성
            if abs(turn_angle) < 15:  # 직진
                direction = "직진"
            elif turn_angle > 45:  # 큰 우회전
                direction = "우회전"
            elif turn_angle > 15:  # 작은 우회전
                direction = "약간 우회전"
            elif turn_angle < -45:  # 큰 좌회전
                direction = "좌회전"
            elif turn_angle < -15:  # 작은 좌회전
                direction = "약간 좌회전"
            else:
                direction = "직진"
            
            # 접근성 레벨 결정
            difficulty = segment['difficulty_score']
            if difficulty < 0.3:
                accessibility_level = "좋음"
                warning_message = None
            elif difficulty < 0.6:
                accessibility_level = "보통"
                warning_message = "약간의 주의 필요"
            else:
                accessibility_level = "어려움"
                warning_message = "휠체어 이동 시 균형 주의 요망"
            
            # 네비게이션 지시사항 생성
            instruction = f"{direction} {segment['distance_meters']:.1f}m"
            
            nav_segment = segment.copy()
            nav_segment.update({
                'segment_number': i + 1,
                'turn_angle': turn_angle,
                'navigation_instruction': instruction,
                'warning_message': warning_message,
                'accessibility_level': accessibility_level
            })
            
            navigation_segments.append(nav_segment)
            prev_bearing = segment['bearing_degrees']
        
        return navigation_segments
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """두 GPS 좌표 간 방향각 계산 (도)"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        return (bearing_deg + 360) % 360  # 0-360도로 정규화
    
    def _calculate_turn_angle(self, prev_bearing: float, curr_bearing: float) -> float:
        """회전 각도 계산"""
        angle_diff = curr_bearing - prev_bearing
        
        # -180 ~ 180도 범위로 정규화
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
            
        return angle_diff
    
    def _analyze_difficulty(self, segments: List[Dict], gps_data: pd.DataFrame) -> Dict[str, Any]:
        """전체 간선 난이도 분석"""
        if not segments:
            return {
                'total_segments': 0,
                'weighted_difficulty': 0.0,
                'difficulty_level': '쉬움',
                'difficulty_grade': 0,
                'cluster_ratios': {0: 1.0, 1: 0.0, 2: 0.0},
                'avg_segment_difficulty': 0.0
            }
        
        # 세그먼트별 난이도 점수 수집
        difficulty_scores = [seg['difficulty_score'] for seg in segments]
        avg_difficulty = np.mean(difficulty_scores)
        
        # 클러스터 분류 (간단한 임계값 기반)
        cluster_labels = []
        for score in difficulty_scores:
            if score < 0.33:
                cluster_labels.append(0)  # 쉬움
            elif score < 0.66:
                cluster_labels.append(1)  # 보통
            else:
                cluster_labels.append(2)  # 어려움
        
        # 클러스터 비율 계산
        total_segments = len(segments)
        cluster_ratios = {
            0: cluster_labels.count(0) / total_segments,
            1: cluster_labels.count(1) / total_segments,
            2: cluster_labels.count(2) / total_segments
        }
        
        # 전체 난이도 결정
        if avg_difficulty < 0.33:
            difficulty_level = '쉬움'
            difficulty_grade = 0
        elif avg_difficulty < 0.66:
            difficulty_level = '보통'
            difficulty_grade = 1
        else:
            difficulty_level = '어려움'
            difficulty_grade = 2
        
        # 세그먼트에 클러스터 라벨 추가
        for i, segment in enumerate(segments):
            segment['cluster_label'] = cluster_labels[i] if i < len(cluster_labels) else 0
        
        return {
            'total_segments': total_segments,
            'weighted_difficulty': avg_difficulty,
            'difficulty_level': difficulty_level,
            'difficulty_grade': difficulty_grade,
            'cluster_ratios': cluster_ratios,
            'avg_segment_difficulty': avg_difficulty
        }
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """두 GPS 좌표 간 거리 계산 (미터)"""
        import math
        
        R = 6371000  # 지구 반지름
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def process_all_sessions(self) -> bool:
        """모든 세션 데이터 처리 및 DB 저장"""
        logger.info("🚀 모든 세션 데이터 처리 시작")
        
        # 노드 데이터 로드
        self.load_nodes()
        
        # 매핑된 모든 세션 폴더 찾기
        target_folders = []
        session_numbers = list(self.edge_mappings.keys())
        
        for session_num in session_numbers:
            # 폴더명 패턴 찾기
            found_folder = None
            for item in os.listdir('.'):
                if os.path.isdir(item) and item.startswith(session_num):
                    location_file = Path(item) / 'Location.csv'
                    if location_file.exists():
                        found_folder = item
                        break
            
            if found_folder:
                target_folders.append(found_folder)
                logger.info(f"   📁 세션 {session_num}: {found_folder}")
            else:
                logger.warning(f"   ⚠️  세션 {session_num} 폴더를 찾을 수 없습니다.")
        
        logger.info(f"📂 총 {len(target_folders)}개 세션 폴더 발견")
        
        # 각 폴더별 데이터 분석
        success_count = 0
        for folder in target_folders:
            try:
                edge_data = self.analyze_edge_data(folder)
                if edge_data:
                    self.edges_data[edge_data['edge_id']] = edge_data
                    success_count += 1
                    logger.info(f"   ✅ {folder} 분석 완료")
            except Exception as e:
                logger.error(f"   ❌ {folder} 분석 실패: {e}")
        
        if not self.edges_data:
            logger.error("❌ 분석할 간선 데이터가 없습니다.")
            return False
        
        logger.info(f"✅ 세션 분석 완료: {success_count}/{len(target_folders)}개 성공")
        
        # DB 저장
        return self.save_to_database()
    
    def save_to_database(self) -> bool:
        """분석 결과를 데이터베이스에 저장"""
        logger.info("💾 데이터베이스 저장 시작...")
        
        if not self.db_manager.connect():
            logger.error("❌ 데이터베이스 연결 실패")
            return False
        
        try:
            # 1. 난이도 클러스터 초기화
            self.db_manager.initialize_difficulty_clusters()
            
            # 2. 노드 저장
            if not self.db_manager.save_nodes(self.nodes_data):
                raise Exception("노드 저장 실패")
            
            # 3. 엣지 저장
            if not self.db_manager.save_edges(self.edges_data):
                raise Exception("엣지 저장 실패")
            
            # 4. 세그먼트 저장 (네비게이션 세그먼트 포함)
            if not self.db_manager.save_segments_with_navigation(self.edges_data):
                raise Exception("세그먼트 저장 실패")
            
            logger.info("🎉 데이터베이스 저장 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 저장 중 오류: {e}")
            return False
        
        finally:
            self.db_manager.disconnect()
    
    def print_summary(self):
        """분석 결과 요약 출력"""
        if not self.edges_data:
            logger.info("분석된 데이터가 없습니다.")
            return
        
        logger.info("\n📊 간선별 분석 결과 요약:")
        logger.info("=" * 60)
        
        for edge_id, edge_data in self.edges_data.items():
            analysis = edge_data['difficulty_analysis']
            gps_analysis = edge_data['gps_analysis']
            
            logger.info(f"\n🛤️  {edge_id}:")
            logger.info(f"   📍 경로: {edge_data['from_node']} → {edge_data['to_node']}")
            logger.info(f"   📊 GPS 포인트: {gps_analysis['total_points']}개")
            logger.info(f"   ⏱️  소요시간: {gps_analysis['duration']:.1f}초")
            logger.info(f"   📏 거리: {gps_analysis.get('path_distance', 0):.1f}m")
            logger.info(f"   🎯 세그먼트: {analysis['total_segments']}개")
            logger.info(f"   🔴 난이도: {analysis['difficulty_level']} ({analysis['weighted_difficulty']:.3f})")
            
            # 클러스터 분포
            ratios = analysis['cluster_ratios']
            logger.info(f"   📈 분포: 쉬움 {ratios.get(0, ratios.get('0', 0)):.1%}, 보통 {ratios.get(1, ratios.get('1', 0)):.1%}, 어려움 {ratios.get(2, ratios.get('2', 0)):.1%}")
            
            # 네비게이션 세그먼트 정보
            nav_segments = edge_data.get('navigation_segments', [])
            if nav_segments:
                logger.info(f"   🗺️  네비게이션: {len(nav_segments)}개 안내 구간")
                # 첫 3개 안내 메시지 표시
                for i, seg in enumerate(nav_segments[:3]):
                    instruction = seg.get('navigation_instruction', '정보없음')
                    accessibility = seg.get('accessibility_level', '알수없음')
                    logger.info(f"      {i+1}. {instruction} (접근성: {accessibility})")

def main():
    """메인 실행 함수"""
    print("🚀 캠퍼스 네비게이션 세션 데이터 DB 저장 시스템")
    print("=" * 60)
    
    try:
        # 데이터 분석 및 저장
        saver = EdgeDataSaver()
        
        if saver.process_all_sessions():
            saver.print_summary()
            print("\n✅ 모든 작업 완료!")
        else:
            print("\n❌ 작업 실패")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()