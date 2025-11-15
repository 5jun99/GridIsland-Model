#!/usr/bin/env python3
"""
데이터베이스 관리 클래스 - 분석 결과를 DB에 저장
"""

import mysql.connector
from mysql.connector import Error
import json
import math
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class DatabaseManager:
    """난이도 분석 결과를 데이터베이스에 저장하는 클래스"""
    
    def __init__(self, host: str = 'localhost', database: str = 'grid_island', 
                 user: str = 'root', password: str = '', port: int = 3306):
        """
        Args:
            host: MySQL 호스트
            database: 데이터베이스 이름
            user: 사용자명
            password: 비밀번호
            port: 포트번호
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """데이터베이스 연결"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
                charset='utf8mb4',
                autocommit=False
            )
            
            if self.connection.is_connected():
                self.logger.info(f"✅ MySQL 연결 성공: {self.database}")
                return True
                
        except Error as e:
            self.logger.error(f"❌ MySQL 연결 실패: {e}")
            return False
    
    def disconnect(self):
        """데이터베이스 연결 해제"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            self.logger.info("🔌 MySQL 연결 해제")
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False) -> Any:
        """SQL 쿼리 실행"""
        if not self.connection or not self.connection.is_connected():
            self.logger.error("❌ 데이터베이스가 연결되지 않았습니다.")
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params or ())
            
            if fetch:
                result = cursor.fetchall()
                cursor.close()
                return result
            else:
                cursor.close()
                return cursor.rowcount
                
        except Error as e:
            self.logger.error(f"❌ 쿼리 실행 실패: {e}")
            self.logger.error(f"쿼리: {query}")
            self.logger.error(f"파라미터: {params}")
            return None
    
    def save_nodes(self, nodes: Dict[str, Dict]) -> bool:
        """노드 데이터 저장"""
        self.logger.info("💾 노드 데이터 저장 중...")
        
        # 기존 노드 데이터 삭제 대신 업데이트 방식 사용
        # delete_query = "DELETE FROM nodes"
        # self.execute_query(delete_query)
        
        upsert_query = """
        INSERT INTO nodes (
            node_id, latitude, longitude, node_name, node_type,
            matched_gps_index, match_distance
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            latitude=VALUES(latitude),
            longitude=VALUES(longitude),
            node_name=VALUES(node_name),
            node_type=VALUES(node_type),
            matched_gps_index=VALUES(matched_gps_index),
            match_distance=VALUES(match_distance)
        """
        
        saved_count = 0
        for node_id, node_info in nodes.items():
            params = (
                node_id,
                float(node_info['latitude']),
                float(node_info['longitude']),
                node_info.get('name', node_id),
                node_info.get('type', 'waypoint'),
                node_info.get('best_match_idx'),
                node_info.get('best_match_distance')
            )
            
            if self.execute_query(upsert_query, params):
                saved_count += 1
        
        self.connection.commit()
        self.logger.info(f"✅ 노드 저장 완료: {saved_count}개")
        return True  # 노드 처리가 완료되면 성공으로 간주
    
    def save_edges(self, edges: Dict[str, Dict]) -> bool:
        """엣지 데이터 저장"""
        self.logger.info("💾 엣지 데이터 저장 중...")
        
        # 기존 세그먼트 데이터 먼저 삭제 (외래키 제약)
        delete_segments_query = "DELETE FROM segments WHERE edge_id IN (SELECT edge_id FROM edges)"
        self.execute_query(delete_segments_query)
        
        # 기존 엣지 데이터 삭제
        delete_query = "DELETE FROM edges"
        self.execute_query(delete_query)
        
        insert_query = """
        INSERT INTO edges (
            edge_id, from_node_id, to_node_id, start_gps_index, end_gps_index,
            path_distance, path_duration, total_segments, difficulty_score,
            difficulty_level, difficulty_grade, cluster_distribution, avg_segment_difficulty
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        saved_count = 0
        for edge_id, edge_info in edges.items():
            # 난이도 분석 결과가 있는 경우만 저장
            if 'difficulty_analysis' not in edge_info:
                continue
                
            analysis = edge_info['difficulty_analysis']
            
            # GPS 데이터에서 경로 정보 추출
            gps_data = edge_info.get('gps_data')
            path_distance = 0
            path_duration = 0
            
            if gps_data is not None and len(gps_data) > 0:
                # 거리 계산 (간단히 직선거리로 근사) - 컬럼명이 이미 정리되어 있음
                start_lat, start_lng = gps_data.iloc[0]['latitude'], gps_data.iloc[0]['longitude']
                end_lat, end_lng = gps_data.iloc[-1]['latitude'], gps_data.iloc[-1]['longitude']
                path_distance = self._calculate_distance(start_lat, start_lng, end_lat, end_lng)
                
                # 시간 계산
                path_duration = gps_data['time_s'].max() - gps_data['time_s'].min()
            
            params = (
                edge_id,
                edge_info['from_node'],
                edge_info['to_node'],
                edge_info.get('start_idx'),
                edge_info.get('end_idx'),
                float(path_distance),
                float(path_duration),
                analysis['total_segments'],
                float(analysis['weighted_difficulty']),
                analysis['difficulty_level'],
                analysis['difficulty_grade'],
                json.dumps(analysis['cluster_ratios']),
                float(analysis['avg_segment_difficulty'])
            )
            
            if self.execute_query(insert_query, params):
                saved_count += 1
        
        self.connection.commit()
        self.logger.info(f"✅ 엣지 저장 완료: {saved_count}개")
        return saved_count > 0
    
    def save_segments(self, edges: Dict[str, Dict]) -> bool:
        """세그먼트 데이터 저장"""
        self.logger.info("💾 세그먼트 데이터 저장 중...")
        
        # 기존 세그먼트 데이터 삭제
        delete_query = "DELETE FROM segments"
        self.execute_query(delete_query)
        
        insert_query = """
        INSERT INTO segments (
            edge_id, segment_number, start_time, end_time, duration,
            vibration_rms, vibration_std, vibration_max,
            rotation_mean, rotation_std, rotation_max,
            height_change, velocity_mean, velocity_std,
            cluster_label, difficulty_score
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        saved_count = 0
        for edge_id, edge_info in edges.items():
            if 'segments' not in edge_info:
                continue
            
            for segment in edge_info['segments']:
                duration = segment['end_time'] - segment['start_time']
                
                params = (
                    edge_id,
                    segment['segment_id'],
                    float(segment['start_time']),
                    float(segment['end_time']),
                    float(duration),
                    float(segment['vibration_rms']),
                    float(segment['vibration_std']),
                    float(segment['vibration_max']),
                    float(segment['rotation_mean']),
                    float(segment['rotation_std']),
                    float(segment['rotation_max']),
                    float(segment['height_change']),
                    float(segment['velocity_mean']),
                    float(segment['velocity_std']),
                    segment.get('cluster_label'),
                    float(segment['difficulty_score'])
                )
                
                if self.execute_query(insert_query, params):
                    saved_count += 1
        
        self.connection.commit()
        self.logger.info(f"✅ 세그먼트 저장 완료: {saved_count}개")
        
        return saved_count > 0
    
    def save_segments_with_navigation(self, edges: Dict[str, Dict]) -> bool:
        """네비게이션 세그먼트 데이터 저장 (확장된 스키마 지원)"""
        self.logger.info("🗺️ 네비게이션 세그먼트 저장 중...")
        
        # 기존 세그먼트 데이터 삭제
        delete_query = "DELETE FROM segments"
        self.execute_query(delete_query)
        
        # 네비게이션 필드가 있는지 확인
        check_nav_fields = """
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'segments' 
        AND COLUMN_NAME IN ('start_lat', 'navigation_instruction')
        """
        
        cursor = self.connection.cursor()
        cursor.execute(check_nav_fields, (self.database,))
        nav_columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        has_navigation_fields = 'start_lat' in nav_columns and 'navigation_instruction' in nav_columns
        
        if has_navigation_fields:
            # 확장된 네비게이션 필드와 함께 저장
            insert_query = """
            INSERT INTO segments (
                edge_id, segment_number, start_time, end_time, duration,
                vibration_rms, vibration_std, vibration_max,
                rotation_mean, rotation_std, rotation_max,
                height_change, velocity_mean, velocity_std,
                cluster_label, difficulty_score,
                start_lat, start_lon, end_lat, end_lon,
                distance_meters, bearing_degrees, turn_angle,
                navigation_instruction, warning_message, estimated_time_sec,
                accessibility_level, is_merged, original_segment_ids
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        else:
            # 기본 필드만으로 저장
            insert_query = """
            INSERT INTO segments (
                edge_id, segment_number, start_time, end_time, duration,
                vibration_rms, vibration_std, vibration_max,
                rotation_mean, rotation_std, rotation_max,
                height_change, velocity_mean, velocity_std,
                cluster_label, difficulty_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.logger.warning("⚠️ 네비게이션 필드가 없는 기존 스키마 사용")
        
        saved_count = 0
        for edge_id, edge_info in edges.items():
            # 네비게이션 세그먼트가 있으면 우선 사용, 없으면 일반 세그먼트 사용
            segments_to_save = edge_info.get('navigation_segments', edge_info.get('segments', []))
            
            for segment in segments_to_save:
                duration = segment['end_time'] - segment['start_time']
                
                # 기본 파라미터
                base_params = (
                    edge_id,
                    segment.get('segment_number', segment.get('segment_id', 1)),
                    float(segment['start_time']),
                    float(segment['end_time']),
                    float(duration),
                    float(segment.get('vibration_rms', 0)),
                    float(segment.get('vibration_std', 0)),
                    float(segment.get('vibration_max', 0)),
                    float(segment.get('rotation_mean', 0)),
                    float(segment.get('rotation_std', 0)),
                    float(segment.get('rotation_max', 0)),
                    float(segment.get('height_change', 0)),
                    float(segment.get('velocity_mean', 1.0)),
                    float(segment.get('velocity_std', 0.1)),
                    segment.get('cluster_label', 0),
                    float(segment.get('difficulty_score', 0.5))
                )
                
                if has_navigation_fields:
                    # 네비게이션 파라미터 추가
                    nav_params = (
                        float(segment.get('start_lat', 37.5665)),
                        float(segment.get('start_lon', 126.9780)),
                        float(segment.get('end_lat', 37.5665)),
                        float(segment.get('end_lon', 126.9780)),
                        float(segment.get('distance_meters', 10.0)),
                        float(segment.get('bearing_degrees', 0.0)),
                        float(segment.get('turn_angle', 0.0)),
                        segment.get('navigation_instruction', '직진'),
                        segment.get('warning_message'),
                        float(segment.get('estimated_time_sec', duration)),
                        segment.get('accessibility_level', '보통'),
                        segment.get('is_merged', False),
                        json.dumps(segment.get('original_segment_ids', [segment.get('segment_id', 1)]))
                    )
                    params = base_params + nav_params
                else:
                    params = base_params
                
                if self.execute_query(insert_query, params):
                    saved_count += 1
        
        self.connection.commit()
        self.logger.info(f"✅ {'네비게이션' if has_navigation_fields else '기본'} 세그먼트 저장 완료: {saved_count}개")
        return saved_count > 0
    
    def save_gps_tracks(self, edges: Dict[str, Dict]) -> bool:
        """GPS 트랙 데이터 저장 (선택사항)"""
        self.logger.info("💾 GPS 트랙 데이터 저장 중...")
        
        # 기존 GPS 데이터 삭제
        delete_query = "DELETE FROM gps_tracks"
        self.execute_query(delete_query)
        
        insert_query = """
        INSERT INTO gps_tracks (
            edge_id, gps_index, timestamp_sec, latitude, longitude, height, velocity
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        saved_count = 0
        for edge_id, edge_info in edges.items():
            gps_data = edge_info.get('gps_data')
            if gps_data is None or len(gps_data) == 0:
                continue
            
            for idx, row in gps_data.iterrows():
                params = (
                    edge_id,
                    int(idx),
                    float(row['time_s']),
                    float(row['latitude']),
                    float(row['longitude']),
                    float(row['height_m']),
                    float(row.get('velocity_ms', 0))
                )
                
                if self.execute_query(insert_query, params):
                    saved_count += 1
        
        self.connection.commit()
        self.logger.info(f"✅ GPS 트랙 저장 완료: {saved_count}개")
        return saved_count > 0
    
    def initialize_difficulty_clusters(self) -> bool:
        """난이도 클러스터 초기 데이터 설정"""
        self.logger.info("🎯 난이도 클러스터 초기화...")
        
        # 기존 데이터 확인
        check_query = "SELECT COUNT(*) FROM difficulty_clusters"
        result = self.execute_query(check_query, fetch=True)
        
        if result and result[0][0] > 0:
            self.logger.info("이미 난이도 클러스터 데이터가 존재합니다.")
            return True
        
        # 초기 데이터 삽입
        insert_query = """
        INSERT INTO difficulty_clusters 
        (cluster_id, cluster_name, color_code, difficulty_range_min, difficulty_range_max, description) 
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        clusters = [
            (0, '쉬움', '#green', 0.0, 0.33, '진동과 회전이 적고 평탄한 구간'),
            (1, '보통', '#orange', 0.33, 0.66, '중간 수준의 진동과 회전이 있는 구간'),
            (2, '어려움', '#red', 0.66, 1.0, '진동과 회전이 심하고 험난한 구간')
        ]
        
        saved_count = 0
        for cluster in clusters:
            if self.execute_query(insert_query, cluster):
                saved_count += 1
        
        self.connection.commit()
        self.logger.info(f"✅ 난이도 클러스터 초기화 완료: {saved_count}개")
        return saved_count > 0
    
    def save_analysis_results(self, analyzer) -> bool:
        """전체 분석 결과를 DB에 저장"""
        self.logger.info("🚀 전체 분석 결과 DB 저장 시작...")
        
        if not self.connect():
            return False
        
        try:
            # 1. 난이도 클러스터 초기화
            self.initialize_difficulty_clusters()
            
            # 2. 노드 저장
            if not self.save_nodes(analyzer.nodes):
                raise Exception("노드 저장 실패")
            
            # 3. 엣지 저장
            if not self.save_edges(analyzer.edges):
                raise Exception("엣지 저장 실패")
            
            # 4. 세그먼트 저장 (네비게이션 정보 포함)
            if not self.save_navigation_segments(analyzer.edges):
                raise Exception("네비게이션 세그먼트 저장 실패")
            
            # 5. GPS 트랙 저장 (선택사항)
            # self.save_gps_tracks(analyzer.edges)
            
            self.logger.info("🎉 전체 분석 결과 DB 저장 완료!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ DB 저장 중 오류: {e}")
            if self.connection:
                self.connection.rollback()
            return False
        
        finally:
            self.disconnect()
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """두 GPS 좌표 간 거리 계산 (미터)"""
        import math
        
        R = 6371000  # 지구 반지름 (미터)
        
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
    
    def merge_similar_segments(self, segments_data: List[Dict]) -> List[Dict]:
        """연속된 비슷한 난이도 구간을 병합"""
        if not segments_data:
            return []
        
        self.logger.info(f"🔄 세그먼트 병합 시작: {len(segments_data)}개 원본 세그먼트")
        
        merged_segments = []
        current_segment = None
        
        # 세그먼트를 시간순으로 정렬
        segments_data = sorted(segments_data, key=lambda x: x['start_time'])
        
        for segment in segments_data:
            if current_segment is None:
                current_segment = segment.copy()
                current_segment['original_segment_ids'] = [segment.get('segment_id', segment['segment_number'])]
                continue
            
            # 연속성 확인 (시간 간격이 10초 이내)
            time_gap = abs(segment['start_time'] - current_segment['end_time'])
            
            # 난이도 차이 확인
            diff_score = abs(segment['difficulty_score'] - current_segment['difficulty_score'])
            
            # 같은 클러스터이고 난이도가 비슷하며 연속된 구간이면 병합
            if (segment['cluster_label'] == current_segment['cluster_label'] and 
                diff_score <= 0.15 and  # 난이도 차이 임계값
                time_gap <= 10.0):  # 시간 간격 임계값
                
                # 세그먼트 병합
                current_segment = self._merge_two_segments(current_segment, segment)
                current_segment['original_segment_ids'].append(
                    segment.get('segment_id', segment['segment_number'])
                )
            else:
                # 현재 세그먼트를 저장하고 새 세그먼트 시작
                current_segment['is_merged'] = len(current_segment['original_segment_ids']) > 1
                merged_segments.append(current_segment)
                
                current_segment = segment.copy()
                current_segment['original_segment_ids'] = [segment.get('segment_id', segment['segment_number'])]
        
        # 마지막 세그먼트 처리
        if current_segment:
            current_segment['is_merged'] = len(current_segment['original_segment_ids']) > 1
            merged_segments.append(current_segment)
        
        self.logger.info(f"✅ 세그먼트 병합 완료: {len(merged_segments)}개 병합 세그먼트")
        return merged_segments
    
    def _merge_two_segments(self, seg1: Dict, seg2: Dict) -> Dict:
        """두 세그먼트를 병합"""
        merged = seg1.copy()
        
        # 시간 범위 확장
        merged['start_time'] = min(seg1['start_time'], seg2['start_time'])
        merged['end_time'] = max(seg1['end_time'], seg2['end_time'])
        merged['duration'] = merged['end_time'] - merged['start_time']
        
        # 평균값 계산
        total_duration = seg1['duration'] + seg2['duration']
        weight1 = seg1['duration'] / total_duration if total_duration > 0 else 0.5
        weight2 = seg2['duration'] / total_duration if total_duration > 0 else 0.5
        
        # 가중평균으로 센서 값들 병합
        sensor_fields = ['vibration_rms', 'vibration_std', 'vibration_max',
                        'rotation_mean', 'rotation_std', 'rotation_max',
                        'height_change', 'velocity_mean', 'velocity_std', 'difficulty_score']
        
        for field in sensor_fields:
            if field in seg1 and field in seg2:
                merged[field] = seg1[field] * weight1 + seg2[field] * weight2
        
        return merged
    
    def calculate_navigation_info(self, gps_data: pd.DataFrame, segment: Dict) -> Dict:
        """GPS 데이터로부터 네비게이션 정보 계산"""
        if gps_data is None or len(gps_data) < 2:
            return {
                'start_lat': None, 'start_lon': None,
                'end_lat': None, 'end_lon': None,
                'distance_meters': 0, 'bearing_degrees': 0,
                'turn_angle': 0, 'estimated_time_sec': 0
            }
        
        # 시간 범위에 해당하는 GPS 포인트 추출
        start_time = segment['start_time']
        end_time = segment['end_time']
        
        # time_s 컬럼을 사용하여 필터링
        segment_gps = gps_data[
            (gps_data['time_s'] >= start_time) & 
            (gps_data['time_s'] <= end_time)
        ].copy()
        
        if len(segment_gps) < 2:
            # 전체 데이터에서 가장 가까운 시간의 포인트 사용
            start_idx = (gps_data['time_s'] - start_time).abs().idxmin()
            end_idx = (gps_data['time_s'] - end_time).abs().idxmin()
            
            if start_idx == end_idx and len(gps_data) > 1:
                end_idx = start_idx + 1 if start_idx < len(gps_data) - 1 else start_idx - 1
            
            start_point = gps_data.loc[start_idx]
            end_point = gps_data.loc[end_idx]
        else:
            start_point = segment_gps.iloc[0]
            end_point = segment_gps.iloc[-1]
        
        # 거리 계산
        distance = self._calculate_distance(
            start_point['latitude'], start_point['longitude'],
            end_point['latitude'], end_point['longitude']
        )
        
        # 방향각 계산 (북쪽 기준)
        bearing = self._calculate_bearing(
            start_point['latitude'], start_point['longitude'],
            end_point['latitude'], end_point['longitude']
        )
        
        # 예상 시간 계산 (난이도 기반 속도 조정)
        base_speed = 1.2  # 기본 속도 m/s
        difficulty_penalty = 1 + segment.get('difficulty_score', 0) * 1.5
        estimated_time = distance / (base_speed / difficulty_penalty) if distance > 0 else 0
        
        return {
            'start_lat': float(start_point['latitude']),
            'start_lon': float(start_point['longitude']),
            'end_lat': float(end_point['latitude']),
            'end_lon': float(end_point['longitude']),
            'distance_meters': distance,
            'bearing_degrees': bearing,
            'turn_angle': 0,  # 이후 경로 연결 시 계산
            'estimated_time_sec': estimated_time
        }
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """두 GPS 좌표 간의 방향각 계산 (북쪽 기준 0도)"""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = (math.cos(lat1) * math.sin(lat2) - 
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360  # 0-360도 범위로 변환
        
        return bearing
    
    def generate_navigation_instruction(self, segment: Dict, prev_bearing: float = None) -> Dict:
        """세그먼트의 네비게이션 안내 생성"""
        distance = segment.get('distance_meters', 0)
        bearing = segment.get('bearing_degrees', 0)
        difficulty = segment.get('difficulty_score', 0)
        cluster_label = segment.get('cluster_label', 0)
        
        # 방향 지시어
        if distance < 5:
            direction_text = ""
        else:
            direction_text = f"{distance:.0f}m"
        
        # 회전 정보 (이전 방향각이 있는 경우)
        turn_instruction = ""
        if prev_bearing is not None:
            turn_angle = bearing - prev_bearing
            if turn_angle > 180:
                turn_angle -= 360
            elif turn_angle < -180:
                turn_angle += 360
            
            segment['turn_angle'] = turn_angle
            
            if abs(turn_angle) < 15:
                turn_instruction = "직진"
            elif 15 <= turn_angle < 45:
                turn_instruction = "약간 우회전"
            elif turn_angle >= 45:
                turn_instruction = "우회전"
            elif -45 < turn_angle <= -15:
                turn_instruction = "약간 좌회전"
            else:
                turn_instruction = "좌회전"
        else:
            turn_instruction = "직진"
        
        # 기본 안내 메시지
        if direction_text:
            instruction = f"{turn_instruction} {direction_text}"
        else:
            instruction = turn_instruction
        
        # 난이도 기반 주의사항
        warnings = []
        accessibility_level = ""
        
        if difficulty < 0.2:
            accessibility_level = "휠체어 이동 용이"
        elif difficulty < 0.4:
            accessibility_level = "휠체어 이동 가능"
        elif difficulty < 0.6:
            accessibility_level = "휠체어 이동 주의"
            warnings.append("약간의 주의 필요")
        elif difficulty < 0.8:
            accessibility_level = "휠체어 이동 어려움"
            warnings.append("험난한 구간")
        else:
            accessibility_level = "휠체어 이동 매우 어려움"
            warnings.append("매우 험난한 구간")
        
        # 센서 기반 구체적 경고
        vibration = segment.get('vibration_rms', 0)
        rotation = segment.get('rotation_std', 0)
        height_change = segment.get('height_change', 0)
        
        if vibration > 4.0:
            warnings.append("노면이 거침")
        if rotation > 1.0:
            warnings.append("균형 주의")
        if abs(height_change) > 2.0:
            if height_change > 0:
                warnings.append("오르막")
            else:
                warnings.append("내리막")
        
        # 최종 메시지 조합
        warning_text = ""
        if warnings:
            warning_text = " - " + ", ".join(warnings)
        
        return {
            'navigation_instruction': instruction,
            'warning_message': warning_text,
            'accessibility_level': accessibility_level,
            'warnings': warnings
        }
    
    def save_navigation_segments(self, edges: Dict[str, Dict]) -> bool:
        """네비게이션 정보가 포함된 세그먼트 저장"""
        self.logger.info("💾 네비게이션 세그먼트 저장 중...")
        
        # 기존 세그먼트 데이터 삭제
        delete_query = "DELETE FROM segments"
        self.execute_query(delete_query)
        
        insert_query = """
        INSERT INTO segments (
            edge_id, segment_number, start_time, end_time, duration,
            vibration_rms, vibration_std, vibration_max,
            rotation_mean, rotation_std, rotation_max,
            height_change, velocity_mean, velocity_std,
            cluster_label, difficulty_score,
            start_lat, start_lon, end_lat, end_lon,
            distance_meters, bearing_degrees, turn_angle,
            navigation_instruction, warning_message, estimated_time_sec,
            accessibility_level, is_merged, original_segment_ids
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                 %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        saved_count = 0
        for edge_id, edge_info in edges.items():
            if 'segments' not in edge_info:
                continue
            
            # 세그먼트 병합
            original_segments = edge_info['segments']
            merged_segments = self.merge_similar_segments(original_segments)
            
            # GPS 데이터
            gps_data = edge_info.get('gps_data')
            prev_bearing = None
            
            for i, segment in enumerate(merged_segments):
                # 네비게이션 정보 계산
                nav_info = self.calculate_navigation_info(gps_data, segment)
                instruction_info = self.generate_navigation_instruction(segment, prev_bearing)
                
                # 회전각 업데이트
                segment.update(nav_info)
                segment.update(instruction_info)
                
                prev_bearing = nav_info['bearing_degrees']
                
                params = (
                    edge_id,
                    i + 1,  # 새로운 세그먼트 번호
                    float(segment['start_time']),
                    float(segment['end_time']),
                    float(segment['duration']),
                    float(segment.get('vibration_rms', 0)),
                    float(segment.get('vibration_std', 0)),
                    float(segment.get('vibration_max', 0)),
                    float(segment.get('rotation_mean', 0)),
                    float(segment.get('rotation_std', 0)),
                    float(segment.get('rotation_max', 0)),
                    float(segment.get('height_change', 0)),
                    float(segment.get('velocity_mean', 0)),
                    float(segment.get('velocity_std', 0)),
                    segment.get('cluster_label'),
                    float(segment['difficulty_score']),
                    nav_info.get('start_lat'),
                    nav_info.get('start_lon'),
                    nav_info.get('end_lat'),
                    nav_info.get('end_lon'),
                    float(nav_info.get('distance_meters', 0)),
                    float(nav_info.get('bearing_degrees', 0)),
                    float(segment.get('turn_angle', 0)),
                    instruction_info.get('navigation_instruction'),
                    instruction_info.get('warning_message'),
                    float(nav_info.get('estimated_time_sec', 0)),
                    instruction_info.get('accessibility_level'),
                    segment.get('is_merged', False),
                    json.dumps(segment.get('original_segment_ids', []))
                )
                
                if self.execute_query(insert_query, params):
                    saved_count += 1
        
        self.connection.commit()
        self.logger.info(f"✅ 네비게이션 세그먼트 저장 완료: {saved_count}개")
        return saved_count > 0

# 사용 예시
if __name__ == "__main__":
    # 데이터베이스 매니저 생성
    db_manager = DatabaseManager(
        host='219.255.242.174',
        database='grid_island',
        user='5jun99',
        password='12341234'
    )
    
    # 분석기 결과와 함께 사용
    # from edge_difficulty_analyzer import EdgeDifficultyAnalyzer
    # analyzer = EdgeDifficultyAnalyzer()
    # ... 분석 실행 ...
    # db_manager.save_analysis_results(analyzer)
    
    print("💾 DatabaseManager 준비 완료!")