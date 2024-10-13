#!/usr/bin/env python3
"""
노드 네트워크 분석기 - 통합된 지역 내 여러 노드에서 최적 경로 탐색
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import networkx as nx
from pathlib import Path
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
class NetworkNode:
    """네트워크 노드 클래스"""
    node_id: int
    latitude: float
    longitude: float
    height: float
    difficulty: float
    measurement_type: str  # 'real_measured', 'simulated', etc.
    sensor_data: Optional[Dict] = None

@dataclass
class NetworkEdge:
    """네트워크 엣지 클래스"""
    from_node: int
    to_node: int
    distance: float
    difficulty: float
    travel_time: float
    traversable: bool = True

class NodeNetworkAnalyzer:
    """통합 노드 네트워크 기반 경로찾기 시스템"""
    
    def __init__(self):
        self.nodes: Dict[int, NetworkNode] = {}
        self.edges: List[NetworkEdge] = []
        self.graph = nx.Graph()
        self.node_counter = 0
        
        # 실제 GPS 데이터 기반 영역 정의
        self.region_bounds = {
            'lat_min': 37.619000,
            'lat_max': 37.621000,
            'lon_min': 127.057000,
            'lon_max': 127.061000
        }
        
    def create_measurement_grid(self, grid_size=20):
        """측정 지점 그리드 생성"""
        print(f"📍 측정 지점 그리드 생성 ({grid_size}x{grid_size})")
        
        # 위도/경도 그리드 생성
        lat_points = np.linspace(
            self.region_bounds['lat_min'], 
            self.region_bounds['lat_max'], 
            grid_size
        )
        lon_points = np.linspace(
            self.region_bounds['lon_min'], 
            self.region_bounds['lon_max'], 
            grid_size
        )
        
        # 각 그리드 포인트를 노드로 생성
        for i, lat in enumerate(lat_points):
            for j, lon in enumerate(lon_points):
                # 고도 시뮬레이션 (실제 지역 특성 반영)
                height = self._simulate_height(lat, lon)
                
                # 지형 기반 난이도 시뮬레이션
                difficulty = self._simulate_difficulty(lat, lon, height)
                
                # 측정 타입 결정
                measurement_type = self._determine_measurement_type(i, j, grid_size)
                
                node = NetworkNode(
                    node_id=self.node_counter,
                    latitude=lat,
                    longitude=lon,
                    height=height,
                    difficulty=difficulty,
                    measurement_type=measurement_type
                )
                
                self.nodes[self.node_counter] = node
                self.node_counter += 1
        
        print(f"✅ {len(self.nodes)}개 측정 노드 생성 완료")
        return len(self.nodes)
    
    def _simulate_height(self, lat, lon):
        """위치 기반 고도 시뮬레이션"""
        # 실제 지역의 지형 특성을 반영한 고도 시뮬레이션
        base_height = 30.0
        
        # 위도/경도에 따른 변화 패턴
        lat_factor = (lat - self.region_bounds['lat_min']) / (
            self.region_bounds['lat_max'] - self.region_bounds['lat_min']
        )
        lon_factor = (lon - self.region_bounds['lon_min']) / (
            self.region_bounds['lon_max'] - self.region_bounds['lon_min']
        )
        
        # 지형 변화 시뮬레이션
        height_variation = (
            np.sin(lat_factor * 4 * np.pi) * 5 +  # 언덕 패턴
            np.cos(lon_factor * 3 * np.pi) * 3 +  # 계곡 패턴
            np.random.normal(0, 1.5)  # 무작위 변화
        )
        
        return max(20.0, min(50.0, base_height + height_variation))
    
    def _simulate_difficulty(self, lat, lon, height):
        """위치 및 고도 기반 난이도 시뮬레이션"""
        # 기본 난이도
        base_difficulty = 0.0
        
        # 고도 변화에 따른 난이도
        height_factor = max(0, (height - 25) / 25)  # 25m 이상에서 난이도 증가
        
        # 특정 지역의 어려운 구간 시뮬레이션
        # 계단 구간 (남동쪽)
        if lat < 37.6195 and lon > 127.0585:
            stairs_difficulty = 0.3 + np.random.uniform(0, 0.2)
        else:
            stairs_difficulty = 0.0
            
        # 울퉁불퉁한 구간 (서쪽)
        if lon < 127.0580:
            rough_difficulty = 0.1 + np.random.uniform(0, 0.1)
        else:
            rough_difficulty = 0.0
        
        total_difficulty = min(1.0, base_difficulty + height_factor * 0.2 + 
                              stairs_difficulty + rough_difficulty)
        
        return total_difficulty
    
    def _determine_measurement_type(self, i, j, grid_size):
        """그리드 위치에 따른 측정 타입 결정"""
        # 중앙 부분은 실제 측정 데이터
        center_range = grid_size // 4
        if (grid_size//2 - center_range <= i <= grid_size//2 + center_range and
            grid_size//2 - center_range <= j <= grid_size//2 + center_range):
            return 'real_measured'
        
        # 가장자리는 시뮬레이션 데이터
        if i < 2 or i >= grid_size-2 or j < 2 or j >= grid_size-2:
            return 'simulated_boundary'
        
        # 나머지는 혼합 데이터
        return 'simulated_normal'
    
    def build_network_graph(self, max_connection_distance=100):
        """노드들을 연결한 네트워크 그래프 구축"""
        print(f"🔗 네트워크 그래프 구축 (최대 연결 거리: {max_connection_distance}m)")
        
        # 모든 노드 페어에 대해 연결성 검사
        node_list = list(self.nodes.values())
        
        for i, node1 in enumerate(node_list):
            for j, node2 in enumerate(node_list[i+1:], i+1):
                distance = self._calculate_distance(
                    node1.latitude, node1.longitude,
                    node2.latitude, node2.longitude
                )
                
                # 최대 연결 거리 내의 노드들만 연결
                if distance <= max_connection_distance:
                    # 고도 차이에 따른 이동 가능성 판단
                    height_diff = abs(node1.height - node2.height)
                    traversable = height_diff <= 10.0  # 10m 이상 고도차는 이동 불가
                    
                    # 평균 난이도 계산
                    avg_difficulty = (node1.difficulty + node2.difficulty) / 2
                    
                    # 이동 시간 계산 (난이도에 따라 속도 조정)
                    base_speed = 1.5  # m/s
                    difficulty_penalty = 1 + avg_difficulty * 2  # 난이도가 높을수록 느려짐
                    travel_time = distance / (base_speed / difficulty_penalty)
                    
                    edge = NetworkEdge(
                        from_node=node1.node_id,
                        to_node=node2.node_id,
                        distance=distance,
                        difficulty=avg_difficulty,
                        travel_time=travel_time,
                        traversable=traversable
                    )
                    
                    self.edges.append(edge)
        
        # NetworkX 그래프 구축
        self.graph = nx.Graph()
        
        # 노드 추가
        for node in self.nodes.values():
            self.graph.add_node(
                node.node_id,
                pos=(node.longitude, node.latitude),
                difficulty=node.difficulty,
                height=node.height,
                measurement_type=node.measurement_type
            )
        
        # 엣지 추가 (이동 가능한 것만)
        for edge in self.edges:
            if edge.traversable:
                self.graph.add_edge(
                    edge.from_node, edge.to_node,
                    distance=edge.distance,
                    difficulty=edge.difficulty,
                    travel_time=edge.travel_time
                )
        
        print(f"✅ 그래프 구축 완료: {len(self.nodes)}개 노드, {len(self.edges)}개 엣지")
        print(f"   이동 가능한 연결: {self.graph.number_of_edges()}개")
        
        return self.graph
    
    def find_optimal_paths(self, start_node_id: int, end_node_id: int):
        """다중 기준 최적 경로 탐색"""
        print(f"\n🎯 최적 경로 탐색: 노드 {start_node_id} → 노드 {end_node_id}")
        
        if start_node_id not in self.graph or end_node_id not in self.graph:
            print("❌ 시작점 또는 도착점이 그래프에 없습니다")
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
            shortest_difficulty = self._calculate_path_difficulty(shortest_path)
            shortest_time = self._calculate_path_time(shortest_path)
            
            results['shortest_distance'] = {
                'path': shortest_path,
                'distance': shortest_distance,
                'difficulty': shortest_difficulty,
                'time': shortest_time,
                'type': '최단거리'
            }
            
            # 2. 최소난이도 경로
            min_difficulty_path = nx.shortest_path(
                self.graph, start_node_id, end_node_id, weight='difficulty'
            )
            min_difficulty_distance = self._calculate_path_distance(min_difficulty_path)
            min_difficulty_difficulty = nx.shortest_path_length(
                self.graph, start_node_id, end_node_id, weight='difficulty'
            )
            min_difficulty_time = self._calculate_path_time(min_difficulty_path)
            
            results['min_difficulty'] = {
                'path': min_difficulty_path,
                'distance': min_difficulty_distance,
                'difficulty': min_difficulty_difficulty,
                'time': min_difficulty_time,
                'type': '최소난이도'
            }
            
            # 3. 최단시간 경로
            fastest_path = nx.shortest_path(
                self.graph, start_node_id, end_node_id, weight='travel_time'
            )
            fastest_distance = self._calculate_path_distance(fastest_path)
            fastest_difficulty = self._calculate_path_difficulty(fastest_path)
            fastest_time = nx.shortest_path_length(
                self.graph, start_node_id, end_node_id, weight='travel_time'
            )
            
            results['fastest'] = {
                'path': fastest_path,
                'distance': fastest_distance,
                'difficulty': fastest_difficulty,
                'time': fastest_time,
                'type': '최단시간'
            }
            
            # 결과 출력
            print("✅ 경로 탐색 완료:")
            for key, result in results.items():
                print(f"   {result['type']}: {len(result['path'])}개 노드, "
                     f"거리 {result['distance']:.1f}m, "
                     f"난이도 {result['difficulty']:.3f}, "
                     f"시간 {result['time']:.1f}초")
            
        except nx.NetworkXNoPath:
            print("❌ 경로를 찾을 수 없습니다")
        
        return results
    
    def create_network_visualization(self, path_results=None, save_path="results/node_network_map.html"):
        """네트워크 전체 시각화"""
        print(f"\n🗺️  네트워크 지도 생성: {save_path}")
        
        # 중심점 계산
        center_lat = np.mean([node.latitude for node in self.nodes.values()])
        center_lon = np.mean([node.longitude for node in self.nodes.values()])
        
        # Folium 지도 생성
        m = folium.Map(location=[center_lat, center_lon], zoom_start=16)
        
        # 노드별 색상 정의 (측정 타입별)
        node_colors = {
            'real_measured': 'red',
            'simulated_normal': 'blue',
            'simulated_boundary': 'gray'
        }
        
        # 모든 노드 표시
        for node in self.nodes.values():
            color = node_colors.get(node.measurement_type, 'green')
            
            # 난이도에 따른 마커 크기
            marker_size = 5 + (node.difficulty * 15)
            
            folium.CircleMarker(
                location=[node.latitude, node.longitude],
                radius=marker_size,
                popup=f"노드 {node.node_id}<br>"
                     f"난이도: {node.difficulty:.3f}<br>"
                     f"고도: {node.height:.1f}m<br>"
                     f"타입: {node.measurement_type}",
                color=color,
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
        
        # 엣지 표시 (샘플링해서 너무 복잡하지 않게)
        edge_sample = self.edges[::max(1, len(self.edges)//100)]  # 최대 100개만
        for edge in edge_sample:
            if edge.traversable:
                node1 = self.nodes[edge.from_node]
                node2 = self.nodes[edge.to_node]
                
                # 난이도에 따른 선 색상
                if edge.difficulty < 0.1:
                    line_color = 'green'
                elif edge.difficulty < 0.3:
                    line_color = 'orange'
                else:
                    line_color = 'red'
                
                folium.PolyLine(
                    locations=[[node1.latitude, node1.longitude],
                              [node2.latitude, node2.longitude]],
                    color=line_color,
                    weight=2,
                    opacity=0.5
                ).add_to(m)
        
        # 최적 경로들 표시 (있는 경우)
        if path_results:
            path_styles = {
                'shortest_distance': {'color': 'blue', 'weight': 6, 'dash': '5,5'},
                'min_difficulty': {'color': 'green', 'weight': 6, 'dash': '10,5'},
                'fastest': {'color': 'purple', 'weight': 6, 'dash': '15,5'}
            }
            
            for path_type, path_info in path_results.items():
                if path_type in path_styles:
                    style = path_styles[path_type]
                    path_coords = []
                    
                    for node_id in path_info['path']:
                        node = self.nodes[node_id]
                        path_coords.append([node.latitude, node.longitude])
                    
                    folium.PolyLine(
                        locations=path_coords,
                        color=style['color'],
                        weight=style['weight'],
                        opacity=0.9,
                        dash_array=style['dash'],
                        popup=f"{path_info['type']}<br>"
                             f"거리: {path_info['distance']:.1f}m<br>"
                             f"난이도: {path_info['difficulty']:.3f}"
                    ).add_to(m)
        
        # 범례 추가
        legend_html = '''
        <div style="position: fixed; 
                   bottom: 50px; left: 50px; width: 300px; height: 250px; 
                   background-color: white; border:2px solid grey; z-index:9999; 
                   font-size:12px; padding: 10px; overflow-y: auto;">
        <h4>📊 노드 네트워크 범례</h4>
        
        <h5>노드 타입</h5>
        <p><i style="color:red">●</i> 실제 측정 데이터</p>
        <p><i style="color:blue">●</i> 시뮬레이션 데이터</p>
        <p><i style="color:gray">●</i> 경계 영역</p>
        
        <h5>경로 타입</h5>
        <p><i style="color:blue">---</i> 최단거리</p>
        <p><i style="color:green">---</i> 최소난이도</p>
        <p><i style="color:purple">---</i> 최단시간</p>
        
        <h5>노드 크기</h5>
        <p>클수록 높은 난이도</p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # 지도 저장
        m.save(save_path)
        print(f"💾 네트워크 지도 저장 완료")
        
        return m
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """두 GPS 좌표 간 거리 계산 (미터)"""
        # 하버사인 공식
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
    
    def analyze_network_statistics(self):
        """네트워크 통계 분석"""
        print("\n📊 네트워크 통계 분석")
        print("=" * 40)
        
        print(f"총 노드 수: {len(self.nodes)}")
        print(f"총 엣지 수: {len(self.edges)}")
        print(f"연결된 엣지 수: {self.graph.number_of_edges()}")
        
        # 난이도 분포
        difficulties = [node.difficulty for node in self.nodes.values()]
        print(f"난이도 분포:")
        print(f"  평균: {np.mean(difficulties):.3f}")
        print(f"  최소: {np.min(difficulties):.3f}")
        print(f"  최대: {np.max(difficulties):.3f}")
        print(f"  표준편차: {np.std(difficulties):.3f}")
        
        # 연결성 분석
        if nx.is_connected(self.graph):
            print("✅ 그래프가 완전히 연결되어 있습니다")
            print(f"평균 경로 길이: {nx.average_shortest_path_length(self.graph):.2f}")
        else:
            print("❌ 그래프가 분리되어 있습니다")
            components = list(nx.connected_components(self.graph))
            print(f"연결 컴포넌트 수: {len(components)}")
            print(f"가장 큰 컴포넌트 크기: {len(max(components, key=len))}")

def main():
    """메인 실행 함수"""
    print("🚀 노드 네트워크 기반 경로찾기 시스템")
    print("=" * 60)
    
    analyzer = NodeNetworkAnalyzer()
    
    # 1. 측정 지점 그리드 생성
    analyzer.create_measurement_grid(grid_size=15)
    
    # 2. 네트워크 그래프 구축
    analyzer.build_network_graph(max_connection_distance=80)
    
    # 3. 네트워크 통계 분석
    analyzer.analyze_network_statistics()
    
    # 4. 샘플 경로 탐색 (왼쪽 아래 → 오른쪽 위)
    start_node = 0  # 첫 번째 노드
    end_node = len(analyzer.nodes) - 1  # 마지막 노드
    
    path_results = analyzer.find_optimal_paths(start_node, end_node)
    
    # 5. 네트워크 시각화
    analyzer.create_network_visualization(path_results)
    
    print("\n✅ 노드 네트워크 분석 완료!")

if __name__ == "__main__":
    main()