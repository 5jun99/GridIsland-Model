#!/usr/bin/env python3
"""
데이터베이스 관리 클래스 - 분석 결과를 DB에 저장
"""

import mysql.connector
from mysql.connector import Error
import json
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
            
            # 4. 세그먼트 저장
            if not self.save_segments(analyzer.edges):
                raise Exception("세그먼트 저장 실패")
            
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