#!/usr/bin/env python3
"""
GPS 좌표와 가장 가까운 노드를 찾는 도구
"""

import pandas as pd
from typing import List, Tuple
import math

class NearestNodeFinder:
    """GPS 좌표와 가장 가까운 노드를 찾는 클래스"""
    
    def __init__(self, node_coord_file: str = "node_coord.txt"):
        """
        Args:
            node_coord_file: 노드 좌표 파일 경로
        """
        self.node_coords = self.load_node_coordinates(node_coord_file)
        print(f"✅ {len(self.node_coords)}개 노드 로드 완료")
    
    def load_node_coordinates(self, file_path: str) -> List[Tuple[float, float]]:
        """노드 좌표 파일 로드"""
        node_coords = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    lat, lng = map(float, line.split(', '))
                    node_coords.append((lat, lng))
        
        return node_coords
    
    def calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float, method: str = 'euclidean') -> float:
        """두 좌표 간의 거리 계산
        
        Args:
            lat1, lng1: 첫 번째 좌표
            lat2, lng2: 두 번째 좌표
            method: 거리 계산 방법 ('euclidean' 또는 'haversine')
        
        Returns:
            거리 (euclidean: 도 단위, haversine: 미터 단위)
        """
        if method == 'euclidean':
            return math.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2)
        
        elif method == 'haversine':
            # Haversine 공식으로 실제 거리 계산 (미터)
            R = 6371000  # 지구 반지름 (미터)
            
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lng = math.radians(lng2 - lng1)
            
            a = (math.sin(delta_lat / 2)**2 + 
                 math.cos(lat1_rad) * math.cos(lat2_rad) * 
                 math.sin(delta_lng / 2)**2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            
            return R * c
        
        else:
            raise ValueError("method는 'euclidean' 또는 'haversine'이어야 합니다")
    
    def find_nearest_node(self, gps_lat: float, gps_lng: float, method: str = 'euclidean') -> Tuple[int, float]:
        """특정 GPS 좌표와 가장 가까운 노드 찾기
        
        Args:
            gps_lat: GPS 위도
            gps_lng: GPS 경도
            method: 거리 계산 방법
        
        Returns:
            (노드 번호, 거리)
        """
        min_distance = float('inf')
        nearest_node_id = -1
        
        for i, (node_lat, node_lng) in enumerate(self.node_coords):
            distance = self.calculate_distance(gps_lat, gps_lng, node_lat, node_lng, method)
            
            if distance < min_distance:
                min_distance = distance
                nearest_node_id = i
        
        return nearest_node_id, min_distance
    
    def find_nearest_nodes_for_gps_data(self, gps_file: str, method: str = 'euclidean') -> pd.DataFrame:
        """GPS 데이터 파일의 모든 좌표에 대해 가장 가까운 노드 찾기
        
        Args:
            gps_file: GPS 데이터 파일 경로
            method: 거리 계산 방법
        
        Returns:
            노드 정보가 추가된 DataFrame
        """
        # GPS 데이터 로드
        gps_data = pd.read_csv(gps_file)
        
        # 컬럼명 정리
        gps_data.rename(columns={
            'Time (s)': 'time_s',
            'Latitude (°)': 'latitude',
            'Longitude (°)': 'longitude',
            'Height (m)': 'height_m',
            'Velocity (m/s)': 'velocity_ms',
            'Direction (°)': 'direction_deg',
            'Horizontal Accuracy (m)': 'h_accuracy_m',
            'Vertical Accuracy (°)': 'v_accuracy_deg'
        }, inplace=True)
        
        # 가장 가까운 노드 찾기
        nearest_nodes = []
        distances = []
        
        print(f"🔍 {len(gps_data)}개 GPS 포인트에 대해 가장 가까운 노드 탐색 중...")
        
        for idx, row in gps_data.iterrows():
            if idx % 100 == 0:
                print(f"   진행률: {idx}/{len(gps_data)}")
            
            lat = row['latitude']
            lng = row['longitude']
            
            if pd.isna(lat) or pd.isna(lng):
                nearest_nodes.append(-1)
                distances.append(float('inf'))
                continue
            
            node_id, distance = self.find_nearest_node(lat, lng, method)
            nearest_nodes.append(node_id)
            distances.append(distance)
        
        # 결과 추가
        gps_data['nearest_node_id'] = nearest_nodes
        gps_data['distance_to_node'] = distances
        
        # 노드 좌표 추가
        gps_data['node_latitude'] = gps_data['nearest_node_id'].apply(
            lambda x: self.node_coords[x][0] if x >= 0 else None
        )
        gps_data['node_longitude'] = gps_data['nearest_node_id'].apply(
            lambda x: self.node_coords[x][1] if x >= 0 else None
        )
        
        print(f"✅ 노드 매핑 완료")
        
        return gps_data
    
    def get_node_statistics(self, gps_with_nodes: pd.DataFrame) -> pd.DataFrame:
        """노드별 통계 계산"""
        node_stats = []
        
        for node_id in range(len(self.node_coords)):
            node_data = gps_with_nodes[gps_with_nodes['nearest_node_id'] == node_id]
            
            if len(node_data) > 0:
                stats = {
                    'node_id': node_id,
                    'node_latitude': self.node_coords[node_id][0],
                    'node_longitude': self.node_coords[node_id][1],
                    'gps_count': len(node_data),
                    'avg_distance': node_data['distance_to_node'].mean(),
                    'min_distance': node_data['distance_to_node'].min(),
                    'max_distance': node_data['distance_to_node'].max(),
                    'time_spent': node_data['time_s'].max() - node_data['time_s'].min() if len(node_data) > 1 else 0
                }
            else:
                stats = {
                    'node_id': node_id,
                    'node_latitude': self.node_coords[node_id][0],
                    'node_longitude': self.node_coords[node_id][1],
                    'gps_count': 0,
                    'avg_distance': None,
                    'min_distance': None,
                    'max_distance': None,
                    'time_spent': 0
                }
            
            node_stats.append(stats)
        
        return pd.DataFrame(node_stats)
    
    def print_node_info(self):
        """모든 노드 정보 출력"""
        print("\n📍 노드 좌표 정보:")
        print("-" * 40)
        for i, (lat, lng) in enumerate(self.node_coords):
            print(f"노드 {i:2d}: ({lat:.6f}, {lng:.6f})")

def main():
    """메인 실행 함수"""
    print("🚀 GPS 좌표-노드 매칭 시스템")
    print("=" * 50)
    
    try:
        # 노드 파인더 초기화
        finder = NearestNodeFinder("node_coord.txt")
        
        # 노드 정보 출력
        finder.print_node_info()
        
        # GPS 데이터 파일 처리
        gps_file = "Sss 2025-10-02 15-53-01/Location.csv"
        
        print(f"\n📱 GPS 데이터 처리: {gps_file}")
        
        # 유클리드 거리로 매칭
        gps_with_nodes = finder.find_nearest_nodes_for_gps_data(gps_file, method='euclidean')
        
        # 결과 저장
        output_file = "gps_with_nearest_nodes.csv"
        gps_with_nodes.to_csv(output_file, index=False)
        print(f"💾 결과 저장: {output_file}")
        
        # 노드별 통계
        node_stats = finder.get_node_statistics(gps_with_nodes)
        stats_file = "node_statistics.csv"
        node_stats.to_csv(stats_file, index=False)
        print(f"💾 노드 통계 저장: {stats_file}")
        
        # 요약 정보 출력
        print(f"\n📊 요약 정보:")
        print(f"   총 GPS 포인트: {len(gps_with_nodes)}")
        print(f"   사용된 노드 수: {len(node_stats[node_stats['gps_count'] > 0])}")
        print(f"   평균 노드 거리: {gps_with_nodes['distance_to_node'].mean():.6f}°")
        
        # 가장 많이 사용된 노드 TOP 5
        top_nodes = node_stats.nlargest(5, 'gps_count')[['node_id', 'gps_count', 'avg_distance']]
        print(f"\n🏆 가장 많이 방문한 노드 TOP 5:")
        for _, row in top_nodes.iterrows():
            if row['gps_count'] > 0:
                print(f"   노드 {int(row['node_id']):2d}: {int(row['gps_count']):4d}회 방문, 평균거리 {row['avg_distance']:.6f}°")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()