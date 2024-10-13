#!/usr/bin/env python3
"""
실제 센서 데이터 기반 노드 네트워크 분석기
GPS 측정 지점들을 클러스터링하여 노드로 생성하고 실제 센서 기반 난이도 계산
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from math import radians, cos, sin, asin, sqrt
import seaborn as sns
from matplotlib import font_manager, rc
import platform

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')
else:  # Linux
    rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False

@dataclass
class SensorNode:
    """실제 센서 데이터 기반 노드"""
    node_id: int
    center_lat: float
    center_lon: float
    center_height: float
    avg_difficulty: float
    measurement_count: int
    gps_points: List[Tuple[float, float, float]]  # (lat, lon, height)
    sensor_features: Dict[str, float]
    cluster_radius: float

@dataclass
class SensorEdge:
    """센서 노드 간 연결"""
    from_node: int
    to_node: int
    distance: float
    avg_difficulty: float
    travel_time: float
    confidence: float  # 연결 신뢰도

class RealSensorNodeAnalyzer:
    """실제 센서 데이터 기반 노드 네트워크 분석기"""
    
    def __init__(self):
        self.nodes: Dict[int, SensorNode] = {}
        self.edges: List[SensorEdge] = []
        self.graph = nx.Graph()
        self.raw_gps_data = None
        self.raw_sensor_data = None
        self.feature_extractor = None
        
    def load_real_sensor_data(self, data_folder="data/Sss 2025-10-02 15-53-01"):
        """실제 센서 데이터 로드"""
        print(f"📊 실제 센서 데이터 로드: {data_folder}")
        
        # GPS 데이터 로드
        location_file = Path(data_folder) / "Location.csv"
        self.raw_gps_data = pd.read_csv(location_file)
        
        # 컬럼명 정리
        self.raw_gps_data.columns = [
            'time', 'latitude', 'longitude', 'height', 
            'velocity', 'direction', 'h_accuracy', 'v_accuracy'
        ]
        
        print(f"✅ GPS 데이터 로드: {len(self.raw_gps_data)}개 측정점")
        
        # 센서 데이터 로드
        acc_file = Path(data_folder) / "Accelerometer.csv"
        gyro_file = Path(data_folder) / "Gyroscope.csv"
        
        acc_data = pd.read_csv(acc_file)
        gyro_data = pd.read_csv(gyro_file)
        
        # 센서 데이터 컬럼명 정리
        acc_data.columns = ['time', 'acc_x', 'acc_y', 'acc_z']
        gyro_data.columns = ['time', 'gyro_x', 'gyro_y', 'gyro_z']
        
        # 센서 데이터 병합
        self.raw_sensor_data = pd.merge(acc_data, gyro_data, on='time', how='outer')
        self.raw_sensor_data = self.raw_sensor_data.sort_values('time').reset_index(drop=True)
        
        print(f"✅ 센서 데이터 로드: {len(self.raw_sensor_data)}개 센서 측정")
        
        # 기본 통계
        print(f"GPS 범위:")
        print(f"  위도: {self.raw_gps_data['latitude'].min():.6f} ~ {self.raw_gps_data['latitude'].max():.6f}")
        print(f"  경도: {self.raw_gps_data['longitude'].min():.6f} ~ {self.raw_gps_data['longitude'].max():.6f}")
        print(f"  고도: {self.raw_gps_data['height'].min():.1f} ~ {self.raw_gps_data['height'].max():.1f}m")
        
        return self.raw_gps_data, self.raw_sensor_data
    
    def create_spatial_clusters(self, cluster_method='kmeans', n_clusters=None, min_cluster_size=5):
        """GPS 지점들을 공간적으로 클러스터링하여 노드 생성"""
        print(f"\n🔍 GPS 지점 클러스터링 ({cluster_method})")
        
        # GPS 좌표 준비
        gps_coords = self.raw_gps_data[['latitude', 'longitude']].values
        
        # 좌표 정규화
        scaler = StandardScaler()
        gps_coords_scaled = scaler.fit_transform(gps_coords)
        
        if cluster_method == 'kmeans':
            # 자동 클러스터 수 결정 (실루엣 점수 기반)
            if n_clusters is None:
                n_clusters = self._find_optimal_clusters(gps_coords_scaled, max_k=15)
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(gps_coords_scaled)
            
        elif cluster_method == 'dbscan':
            # DBSCAN으로 밀도 기반 클러스터링
            eps = 0.1  # 클러스터 반경 (정규화된 좌표 기준)
            clusterer = DBSCAN(eps=eps, min_samples=min_cluster_size)
            cluster_labels = clusterer.fit_predict(gps_coords_scaled)
            
            # 노이즈 포인트(-1) 제거
            valid_mask = cluster_labels != -1
            cluster_labels = cluster_labels[valid_mask]
            self.raw_gps_data = self.raw_gps_data[valid_mask].reset_index(drop=True)
            
            n_clusters = len(set(cluster_labels))
        
        print(f"✅ {n_clusters}개 클러스터 생성")
        
        # 각 클러스터를 노드로 변환
        self.nodes = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = self.raw_gps_data[cluster_mask]
            
            if len(cluster_points) < 3:  # 너무 적은 포인트는 제외
                continue
            
            # 클러스터 중심점 계산
            center_lat = cluster_points['latitude'].mean()
            center_lon = cluster_points['longitude'].mean()
            center_height = cluster_points['height'].mean()
            
            # 클러스터 반경 계산
            distances = []
            for _, point in cluster_points.iterrows():
                dist = self._calculate_distance(
                    center_lat, center_lon, point['latitude'], point['longitude']
                )
                distances.append(dist)
            cluster_radius = np.mean(distances)
            
            # GPS 포인트 리스트
            gps_points = [
                (row['latitude'], row['longitude'], row['height']) 
                for _, row in cluster_points.iterrows()
            ]
            
            # 해당 클러스터의 센서 데이터 기반 난이도 계산
            difficulty, features = self._calculate_cluster_difficulty(cluster_points)
            
            # 노드 생성
            node = SensorNode(
                node_id=cluster_id,
                center_lat=center_lat,
                center_lon=center_lon,
                center_height=center_height,
                avg_difficulty=difficulty,
                measurement_count=len(cluster_points),
                gps_points=gps_points,
                sensor_features=features,
                cluster_radius=cluster_radius
            )
            
            self.nodes[cluster_id] = node
        
        print(f"✅ {len(self.nodes)}개 센서 노드 생성 완료")
        
        # 노드별 정보 출력
        for node_id, node in self.nodes.items():
            print(f"  노드 {node_id}: {node.measurement_count}개 측정점, "
                  f"난이도 {node.avg_difficulty:.3f}, 반경 {node.cluster_radius:.1f}m")
        
        return self.nodes
    
    def _find_optimal_clusters(self, coords, max_k=15):
        """실루엣 점수를 기반으로 최적 클러스터 수 찾기"""
        from sklearn.metrics import silhouette_score
        
        silhouette_scores = []
        k_range = range(2, min(max_k, len(coords)//2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(coords)
            score = silhouette_score(coords, labels)
            silhouette_scores.append(score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        best_score = max(silhouette_scores)
        
        print(f"📊 최적 클러스터 수: {optimal_k} (실루엣 점수: {best_score:.3f})")
        return optimal_k
    
    def _calculate_cluster_difficulty(self, cluster_gps_data):
        """클러스터 내 GPS 지점들의 센서 데이터 기반 난이도 계산"""
        # 해당 GPS 시간대의 센서 데이터 추출
        time_range = (cluster_gps_data['time'].min(), cluster_gps_data['time'].max())
        
        # 센서 데이터에서 해당 시간 범위 추출
        sensor_mask = (
            (self.raw_sensor_data['time'] >= time_range[0]) & 
            (self.raw_sensor_data['time'] <= time_range[1])
        )
        cluster_sensor_data = self.raw_sensor_data[sensor_mask]
        
        if len(cluster_sensor_data) < 5:
            # 센서 데이터가 부족한 경우 기본값
            return 0.1, {'insufficient_data': True}
        
        # 가속도 크기 계산
        acc_magnitude = np.sqrt(
            cluster_sensor_data['acc_x']**2 + 
            cluster_sensor_data['acc_y']**2 + 
            cluster_sensor_data['acc_z']**2
        )
        
        # 자이로스코프 크기 계산
        gyro_magnitude = np.sqrt(
            cluster_sensor_data['gyro_x']**2 + 
            cluster_sensor_data['gyro_y']**2 + 
            cluster_sensor_data['gyro_z']**2
        )
        
        # 휠체어 난이도 특성 추출
        features = {
            'acc_mean': np.mean(acc_magnitude),
            'acc_std': np.std(acc_magnitude),
            'acc_max': np.max(acc_magnitude),
            'gyro_mean': np.mean(gyro_magnitude),
            'gyro_std': np.std(gyro_magnitude),
            'vibration_intensity': np.std(acc_magnitude),
            'stability': 1.0 / (1.0 + np.std(gyro_magnitude)),
            'shock_count': np.sum(acc_magnitude > (np.mean(acc_magnitude) + 2 * np.std(acc_magnitude)))
        }
        
        # 난이도 점수 계산 (0-1 범위)
        # 진동, 불안정성, 충격 횟수를 종합적으로 고려
        vibration_score = min(1.0, features['vibration_intensity'] / 3.0)
        instability_score = 1.0 - features['stability']
        shock_score = min(1.0, features['shock_count'] / len(cluster_sensor_data))
        
        difficulty = (vibration_score * 0.4 + instability_score * 0.4 + shock_score * 0.2)
        difficulty = max(0.0, min(1.0, difficulty))  # 0-1 범위로 제한
        
        return difficulty, features
    
    def build_sensor_network(self, max_connection_distance=100):
        """센서 노드들을 연결한 네트워크 구축"""
        print(f"\n🔗 센서 노드 네트워크 구축 (최대 연결 거리: {max_connection_distance}m)")
        
        node_list = list(self.nodes.values())
        
        for i, node1 in enumerate(node_list):
            for j, node2 in enumerate(node_list[i+1:], i+1):
                distance = self._calculate_distance(
                    node1.center_lat, node1.center_lon,
                    node2.center_lat, node2.center_lon
                )
                
                if distance <= max_connection_distance:
                    # 평균 난이도
                    avg_difficulty = (node1.avg_difficulty + node2.avg_difficulty) / 2
                    
                    # 이동 시간 계산 (난이도에 따른 속도 조정)
                    base_speed = 1.5  # m/s
                    difficulty_penalty = 1 + avg_difficulty * 2
                    travel_time = distance / (base_speed / difficulty_penalty)
                    
                    # 연결 신뢰도 (측정점 수가 많을수록 신뢰도 높음)
                    confidence = min(1.0, (node1.measurement_count + node2.measurement_count) / 20)
                    
                    edge = SensorEdge(
                        from_node=node1.node_id,
                        to_node=node2.node_id,
                        distance=distance,
                        avg_difficulty=avg_difficulty,
                        travel_time=travel_time,
                        confidence=confidence
                    )
                    
                    self.edges.append(edge)
        
        # NetworkX 그래프 구축
        self.graph = nx.Graph()
        
        # 노드 추가
        for node in self.nodes.values():
            self.graph.add_node(
                node.node_id,
                pos=(node.center_lon, node.center_lat),
                difficulty=node.avg_difficulty,
                height=node.center_height,
                measurement_count=node.measurement_count,
                radius=node.cluster_radius
            )
        
        # 엣지 추가
        for edge in self.edges:
            self.graph.add_edge(
                edge.from_node, edge.to_node,
                distance=edge.distance,
                difficulty=edge.avg_difficulty,
                travel_time=edge.travel_time,
                confidence=edge.confidence
            )
        
        print(f"✅ 네트워크 구축 완료: {len(self.nodes)}개 노드, {len(self.edges)}개 연결")
        
        return self.graph
    
    def find_optimal_sensor_paths(self, start_node_id: int, end_node_id: int):
        """실제 센서 데이터 기반 최적 경로 탐색"""
        print(f"\n🎯 센서 기반 최적 경로 탐색: 노드 {start_node_id} → 노드 {end_node_id}")
        
        if start_node_id not in self.graph or end_node_id not in self.graph:
            print("❌ 시작점 또는 도착점이 존재하지 않습니다")
            return {}
        
        results = {}
        
        try:
            # 1. 최단거리 경로
            shortest_path = nx.shortest_path(
                self.graph, start_node_id, end_node_id, weight='distance'
            )
            shortest_distance = nx.shortest_path_length(
                self.graph, start_node_id, end_node_id, weight='distance'
            )
            
            results['shortest_distance'] = {
                'path': shortest_path,
                'distance': shortest_distance,
                'difficulty': self._calculate_path_difficulty(shortest_path),
                'time': self._calculate_path_time(shortest_path),
                'confidence': self._calculate_path_confidence(shortest_path),
                'type': '최단거리'
            }
            
            # 2. 최소난이도 경로 (실제 센서 기반)
            min_difficulty_path = nx.shortest_path(
                self.graph, start_node_id, end_node_id, weight='difficulty'
            )
            
            results['min_difficulty'] = {
                'path': min_difficulty_path,
                'distance': self._calculate_path_distance(min_difficulty_path),
                'difficulty': nx.shortest_path_length(
                    self.graph, start_node_id, end_node_id, weight='difficulty'
                ),
                'time': self._calculate_path_time(min_difficulty_path),
                'confidence': self._calculate_path_confidence(min_difficulty_path),
                'type': '최소난이도'
            }
            
            # 3. 최고신뢰도 경로
            # 신뢰도 가중 거리 = 거리 / 신뢰도
            for edge in self.graph.edges(data=True):
                edge[2]['weighted_distance'] = edge[2]['distance'] / max(0.1, edge[2]['confidence'])
            
            high_confidence_path = nx.shortest_path(
                self.graph, start_node_id, end_node_id, weight='weighted_distance'
            )
            
            results['high_confidence'] = {
                'path': high_confidence_path,
                'distance': self._calculate_path_distance(high_confidence_path),
                'difficulty': self._calculate_path_difficulty(high_confidence_path),
                'time': self._calculate_path_time(high_confidence_path),
                'confidence': self._calculate_path_confidence(high_confidence_path),
                'type': '최고신뢰도'
            }
            
            # 결과 출력
            print("✅ 센서 기반 경로 탐색 완료:")
            for key, result in results.items():
                print(f"   {result['type']}: {len(result['path'])}개 노드, "
                     f"거리 {result['distance']:.1f}m, "
                     f"난이도 {result['difficulty']:.3f}, "
                     f"신뢰도 {result['confidence']:.3f}")
            
        except nx.NetworkXNoPath:
            print("❌ 경로를 찾을 수 없습니다")
        
        return results
    
    def create_sensor_network_visualization(self, path_results=None, save_path="results/real_sensor_network_map.html"):
        """실제 센서 네트워크 시각화"""
        print(f"\n🗺️  실제 센서 네트워크 지도 생성: {save_path}")
        
        # 중심점 계산
        center_lat = np.mean([node.center_lat for node in self.nodes.values()])
        center_lon = np.mean([node.center_lon for node in self.nodes.values()])
        
        # Folium 지도 생성
        m = folium.Map(location=[center_lat, center_lon], zoom_start=17)
        
        # 실제 GPS 측정점들 표시 (작은 점들)
        for _, point in self.raw_gps_data.iterrows():
            folium.CircleMarker(
                location=[point['latitude'], point['longitude']],
                radius=1,
                color='lightblue',
                fill=True,
                fillOpacity=0.3,
                popup=f"GPS 측정점<br>고도: {point['height']:.1f}m"
            ).add_to(m)
        
        # 센서 노드들 표시
        for node in self.nodes.values():
            # 난이도에 따른 색상
            if node.avg_difficulty < 0.2:
                color = 'green'
            elif node.avg_difficulty < 0.5:
                color = 'orange'
            else:
                color = 'red'
            
            # 측정점 수에 따른 크기
            radius = 10 + (node.measurement_count / 2)
            
            folium.CircleMarker(
                location=[node.center_lat, node.center_lon],
                radius=radius,
                popup=f"센서 노드 {node.node_id}<br>"
                     f"측정점: {node.measurement_count}개<br>"
                     f"난이도: {node.avg_difficulty:.3f}<br>"
                     f"클러스터 반경: {node.cluster_radius:.1f}m<br>"
                     f"고도: {node.center_height:.1f}m",
                color=color,
                fill=True,
                fillOpacity=0.8
            ).add_to(m)
            
            # 클러스터 범위 표시
            folium.Circle(
                location=[node.center_lat, node.center_lon],
                radius=node.cluster_radius,
                color=color,
                fill=False,
                opacity=0.3
            ).add_to(m)
        
        # 연결선 표시
        for edge in self.edges:
            node1 = self.nodes[edge.from_node]
            node2 = self.nodes[edge.to_node]
            
            # 신뢰도에 따른 선 굵기
            weight = 1 + (edge.confidence * 4)
            
            # 난이도에 따른 색상
            if edge.avg_difficulty < 0.2:
                line_color = 'green'
            elif edge.avg_difficulty < 0.5:
                line_color = 'orange'
            else:
                line_color = 'red'
            
            folium.PolyLine(
                locations=[[node1.center_lat, node1.center_lon],
                          [node2.center_lat, node2.center_lon]],
                color=line_color,
                weight=weight,
                opacity=0.6,
                popup=f"거리: {edge.distance:.1f}m<br>"
                     f"난이도: {edge.avg_difficulty:.3f}<br>"
                     f"신뢰도: {edge.confidence:.3f}"
            ).add_to(m)
        
        # 최적 경로들 표시
        if path_results:
            path_styles = {
                'shortest_distance': {'color': 'blue', 'weight': 8, 'dash': '5,5'},
                'min_difficulty': {'color': 'green', 'weight': 8, 'dash': '10,5'},
                'high_confidence': {'color': 'purple', 'weight': 8, 'dash': '15,5'}
            }
            
            for path_type, path_info in path_results.items():
                if path_type in path_styles:
                    style = path_styles[path_type]
                    path_coords = []
                    
                    for node_id in path_info['path']:
                        node = self.nodes[node_id]
                        path_coords.append([node.center_lat, node.center_lon])
                    
                    folium.PolyLine(
                        locations=path_coords,
                        color=style['color'],
                        weight=style['weight'],
                        opacity=0.9,
                        dash_array=style['dash'],
                        popup=f"{path_info['type']}<br>"
                             f"거리: {path_info['distance']:.1f}m<br>"
                             f"난이도: {path_info['difficulty']:.3f}<br>"
                             f"신뢰도: {path_info['confidence']:.3f}"
                    ).add_to(m)
        
        # 범례 추가
        legend_html = '''
        <div style="position: fixed; 
                   bottom: 50px; left: 50px; width: 350px; height: 300px; 
                   background-color: white; border:2px solid grey; z-index:9999; 
                   font-size:12px; padding: 10px; overflow-y: auto;">
        <h4>📊 실제 센서 네트워크</h4>
        
        <h5>센서 노드</h5>
        <p><i style="color:green">●</i> 낮은 난이도 (< 0.2)</p>
        <p><i style="color:orange">●</i> 중간 난이도 (0.2-0.5)</p>
        <p><i style="color:red">●</i> 높은 난이도 (> 0.5)</p>
        
        <h5>최적 경로</h5>
        <p><i style="color:blue">---</i> 최단거리</p>
        <p><i style="color:green">---</i> 최소난이도</p>
        <p><i style="color:purple">---</i> 최고신뢰도</p>
        
        <h5>기타</h5>
        <p><i style="color:lightblue">●</i> 원본 GPS 측정점</p>
        <p>노드 크기: 측정점 수에 비례</p>
        <p>연결선 굵기: 신뢰도에 비례</p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # 지도 저장
        m.save(save_path)
        print(f"💾 실제 센서 네트워크 지도 저장 완료")
        
        return m
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """GPS 좌표 간 거리 계산 (하버사인 공식)"""
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # 지구 반지름 (km)
        return c * r * 1000  # 미터로 변환
    
    def _calculate_path_distance(self, path):
        """경로의 총 거리 계산"""
        total_distance = 0
        for i in range(len(path) - 1):
            edge_data = self.graph[path[i]][path[i+1]]
            total_distance += edge_data['distance']
        return total_distance
    
    def _calculate_path_difficulty(self, path):
        """경로의 평균 난이도 계산"""
        difficulties = []
        for i in range(len(path) - 1):
            edge_data = self.graph[path[i]][path[i+1]]
            difficulties.append(edge_data['difficulty'])
        return np.mean(difficulties) if difficulties else 0
    
    def _calculate_path_time(self, path):
        """경로의 총 시간 계산"""
        total_time = 0
        for i in range(len(path) - 1):
            edge_data = self.graph[path[i]][path[i+1]]
            total_time += edge_data['travel_time']
        return total_time
    
    def _calculate_path_confidence(self, path):
        """경로의 평균 신뢰도 계산"""
        confidences = []
        for i in range(len(path) - 1):
            edge_data = self.graph[path[i]][path[i+1]]
            confidences.append(edge_data['confidence'])
        return np.mean(confidences) if confidences else 0

def main():
    """메인 실행 함수"""
    print("🚀 실제 센서 데이터 기반 노드 네트워크 분석")
    print("=" * 60)
    
    analyzer = RealSensorNodeAnalyzer()
    
    # 1. 실제 센서 데이터 로드
    analyzer.load_real_sensor_data()
    
    # 2. GPS 지점들을 클러스터링하여 노드 생성 (더 많은 노드로 테스트)
    analyzer.create_spatial_clusters(cluster_method='kmeans', n_clusters=6)
    
    # 3. 센서 노드 네트워크 구축
    analyzer.build_sensor_network(max_connection_distance=200)
    
    # 4. 여러 경로 조합 테스트
    if len(analyzer.nodes) >= 2:
        node_ids = list(analyzer.nodes.keys())
        
        print(f"\n🧪 경로 탐색 테스트 (총 {len(node_ids)}개 노드)")
        print("=" * 40)
        
        # 모든 노드 조합에 대해 경로 탐색
        all_results = {}
        for i, start in enumerate(node_ids):
            for j, end in enumerate(node_ids):
                if start != end:
                    print(f"\n📍 경로 테스트: 노드 {start} → 노드 {end}")
                    path_results = analyzer.find_optimal_sensor_paths(start, end)
                    all_results[f"{start}_{end}"] = path_results
        
        # 대표 경로로 시각화 (0 → 2)
        if 0 in node_ids and 2 in node_ids:
            representative_results = analyzer.find_optimal_sensor_paths(0, 2)
            analyzer.create_sensor_network_visualization(representative_results)
        else:
            # 첫 번째 가능한 조합으로 시각화
            start_node = node_ids[0]
            end_node = node_ids[-1]
            representative_results = analyzer.find_optimal_sensor_paths(start_node, end_node)
            analyzer.create_sensor_network_visualization(representative_results)
    else:
        print("❌ 노드가 부족하여 경로 탐색을 수행할 수 없습니다")
    
    print("\n✅ 실제 센서 기반 노드 네트워크 분석 완료!")

if __name__ == "__main__":
    main()