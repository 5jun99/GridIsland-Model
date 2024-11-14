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
from pathlib import Path
from database_manager import DatabaseManager
from nearest_node_finder import NearestNodeFinder
import logging

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
        
        # 간선 매핑 정보 (분석 결과 기반)
        self.edge_mappings = {
            '01': {'from_node': 7, 'to_node': 8, 'edge_id': 'edge_7_to_8_forward'},
            '02': {'from_node': 8, 'to_node': 7, 'edge_id': 'edge_8_to_7_backward'},
            '03': {'from_node': 7, 'to_node': 9, 'edge_id': 'edge_7_to_9_forward'},
            '04': {'from_node': 9, 'to_node': 7, 'edge_id': 'edge_9_to_7_backward'},
            '05': {'from_node': 6, 'to_node': 7, 'edge_id': 'edge_6_to_7_forward'},
            '06': {'from_node': 7, 'to_node': 6, 'edge_id': 'edge_7_to_6_backward'},
            '07': {'from_node': 4, 'to_node': 6, 'edge_id': 'edge_4_to_6_forward'},
            '08': {'from_node': 6, 'to_node': 4, 'edge_id': 'edge_6_to_4_backward'}
        }
        
        self.nodes_data = {}
        self.edges_data = {}
        
    def load_nodes(self) -> Dict[str, Dict]:
        """노드 좌표 데이터 로드"""
        logger.info("📍 노드 데이터 로드 중...")
        
        for i, (lat, lng) in enumerate(self.node_finder.node_coords):
            node_id = f"node_{i}"
            self.nodes_data[node_id] = {
                'latitude': lat,
                'longitude': lng,
                'name': f'노드 {i}',
                'type': 'waypoint'
            }
        
        logger.info(f"✅ 노드 로드 완료: {len(self.nodes_data)}개")
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
        
        # 세그먼트 분석 (간단한 시간 기반)
        segments = self._create_time_segments(gps_data, sensor_data)
        
        # 난이도 분석
        difficulty_analysis = self._analyze_difficulty(segments, gps_data)
        
        edge_data = {
            'edge_id': edge_info['edge_id'],
            'from_node': f"node_{edge_info['from_node']}",
            'to_node': f"node_{edge_info['to_node']}",
            'data_folder': data_folder,
            'gps_data': gps_data,
            'sensor_data': sensor_data,
            'gps_analysis': gps_analysis,
            'segments': segments,
            'difficulty_analysis': difficulty_analysis
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
    
    def process_edges_1_2_3(self) -> bool:
        """간선 1, 2, 3 데이터 처리 및 DB 저장"""
        logger.info("🚀 간선 1,2,3 데이터 처리 시작")
        
        # 노드 데이터 로드
        self.load_nodes()
        
        # 간선 1, 2, 3에 해당하는 폴더들
        target_folders = []
        
        # 01-06 폴더 찾기 (간선 1,2,3)
        all_folders = [f for f in os.listdir('.') if os.path.isdir(f) and f.startswith('0')]
        
        for folder in sorted(all_folders):
            if any(folder.startswith(prefix) for prefix in ['01', '02', '03', '04', '05', '06']):
                if Path(folder) / 'Location.csv' in [Path(folder) / f for f in os.listdir(folder)]:
                    target_folders.append(folder)
        
        logger.info(f"📂 대상 폴더: {target_folders}")
        
        # 각 폴더별 데이터 분석
        for folder in target_folders:
            try:
                edge_data = self.analyze_edge_data(folder)
                if edge_data:
                    self.edges_data[edge_data['edge_id']] = edge_data
            except Exception as e:
                logger.error(f"❌ {folder} 분석 실패: {e}")
        
        if not self.edges_data:
            logger.error("❌ 분석할 간선 데이터가 없습니다.")
            return False
        
        logger.info(f"✅ 간선 분석 완료: {len(self.edges_data)}개")
        
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
            
            # 4. 세그먼트 저장
            if not self.db_manager.save_segments(self.edges_data):
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
            logger.info(f"   📈 분포: 쉬움 {ratios[0]:.1%}, 보통 {ratios[1]:.1%}, 어려움 {ratios[2]:.1%}")

def main():
    """메인 실행 함수"""
    print("🚀 간선 1,2,3 데이터 DB 저장 시스템")
    print("=" * 50)
    
    try:
        # 데이터 분석 및 저장
        saver = EdgeDataSaver()
        
        if saver.process_edges_1_2_3():
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