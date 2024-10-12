#!/usr/bin/env python3
"""
GPS 그래프 빌더 - GPS 경로 데이터를 노드-엣지 그래프로 변환
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from math import radians, cos, sin, asin, sqrt
import folium

@dataclass
class GPSNode:
    """GPS 노드"""
    id: int
    latitude: float
    longitude: float
    height: float
    difficulty: float
    velocity: float
    window_id: int
    cluster_id: int = 0

@dataclass
class GPSEdge:
    """GPS 엣지"""
    from_node: int
    to_node: int
    distance_m: float
    travel_time_s: float
    difficulty_cost: float
    height_change: float

class GPSGraphBuilder:
    """GPS 경로를 그래프로 변환하는 클래스"""
    
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.graph = None
        self.data_df = None
    
    def load_synchronized_data(self, file_path: str) -> pd.DataFrame:
        """동기화된 GPS-센서 데이터 로드"""
        print(f"📊 동기화 데이터 로드: {file_path}")
        
        self.data_df = pd.read_csv(file_path)
        print(f"✅ 로드 완료: {len(self.data_df)}개 윈도우")
        
        return self.data_df
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine 공식을 사용한 두 GPS 좌표 간 거리 계산 (미터)"""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371000  # 지구 반지름 (미터)
        
        return c * r
    
    def create_nodes(self) -> List[GPSNode]:
        """GPS 데이터에서 노드 생성"""
        if self.data_df is None:
            raise ValueError("데이터를 먼저 로드하세요")
        
        print("🌍 GPS 노드 생성 중...")
        
        self.nodes = []
        
        for idx, row in self.data_df.iterrows():
            node = GPSNode(
                id=int(row['window_id']),
                latitude=row['lat_mean'],
                longitude=row['lng_mean'],
                height=row['height_mean'],
                difficulty=row['difficulty'],
                velocity=row['velocity_mean'],
                window_id=int(row['window_id']),
                cluster_id=int(row['cluster_id'])
            )
            
            self.nodes.append(node)
        
        print(f"✅ 노드 생성 완료: {len(self.nodes)}개")
        
        return self.nodes
    
    def create_edges(self, max_distance_m: float = 100.0) -> List[GPSEdge]:
        """노드 간 엣지 생성"""
        if not self.nodes:
            raise ValueError("노드를 먼저 생성하세요")
        
        print(f"🔗 엣지 생성 중 (최대 거리: {max_distance_m}m)...")
        
        self.edges = []
        
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i >= j:  # 중복 및 자기 자신 제외
                    continue
                
                # 거리 계산
                distance = self.haversine_distance(
                    node1.latitude, node1.longitude,
                    node2.latitude, node2.longitude
                )
                
                # 거리 제한 확인
                if distance <= max_distance_m:
                    # 이동 시간 계산 (평균 속도 사용)
                    avg_velocity = (node1.velocity + node2.velocity) / 2
                    if avg_velocity > 0:
                        travel_time = distance / avg_velocity
                    else:
                        travel_time = distance / 1.0  # 기본 속도 1m/s
                    
                    # 난이도 기반 비용 계산
                    avg_difficulty = (node1.difficulty + node2.difficulty) / 2
                    difficulty_cost = distance * (1 + avg_difficulty * 2)  # 난이도에 따라 비용 증가
                    
                    # 고도 변화
                    height_change = abs(node2.height - node1.height)
                    
                    # 양방향 엣지 생성
                    edge1 = GPSEdge(
                        from_node=node1.id,
                        to_node=node2.id,
                        distance_m=distance,
                        travel_time_s=travel_time,
                        difficulty_cost=difficulty_cost,
                        height_change=height_change
                    )
                    
                    edge2 = GPSEdge(
                        from_node=node2.id,
                        to_node=node1.id,
                        distance_m=distance,
                        travel_time_s=travel_time,
                        difficulty_cost=difficulty_cost,
                        height_change=height_change
                    )
                    
                    self.edges.extend([edge1, edge2])
        
        print(f"✅ 엣지 생성 완료: {len(self.edges)}개")
        
        return self.edges
    
    def build_networkx_graph(self, cost_type: str = 'difficulty') -> nx.Graph:
        """NetworkX 그래프 생성"""
        if not self.nodes or not self.edges:
            raise ValueError("노드와 엣지를 먼저 생성하세요")
        
        print(f"📈 NetworkX 그래프 생성 (비용 유형: {cost_type})...")
        
        self.graph = nx.Graph()
        
        # 노드 추가
        for node in self.nodes:
            self.graph.add_node(node.id, 
                              lat=node.latitude,
                              lng=node.longitude,
                              height=node.height,
                              difficulty=node.difficulty,
                              velocity=node.velocity,
                              cluster_id=node.cluster_id)
        
        # 엣지 추가 (중복 제거를 위해 방향성 없는 그래프 사용)
        added_edges = set()
        
        for edge in self.edges:
            edge_key = tuple(sorted([edge.from_node, edge.to_node]))
            
            if edge_key not in added_edges:
                # 비용 유형에 따라 가중치 선택
                if cost_type == 'distance':
                    weight = edge.distance_m
                elif cost_type == 'difficulty':
                    weight = edge.difficulty_cost
                elif cost_type == 'time':
                    weight = edge.travel_time_s
                else:
                    weight = edge.difficulty_cost
                
                self.graph.add_edge(edge.from_node, edge.to_node,
                                   distance=edge.distance_m,
                                   travel_time=edge.travel_time_s,
                                   difficulty_cost=edge.difficulty_cost,
                                   height_change=edge.height_change,
                                   weight=weight)
                
                added_edges.add(edge_key)
        
        print(f"✅ 그래프 생성 완료: {self.graph.number_of_nodes()}개 노드, {self.graph.number_of_edges()}개 엣지")
        
        return self.graph
    
    def find_optimal_path(self, start_node: int, end_node: int, cost_type: str = 'difficulty') -> Dict:
        """최적 경로 탐색"""
        if self.graph is None:
            raise ValueError("그래프를 먼저 생성하세요")
        
        print(f"🎯 최적 경로 탐색: {start_node} → {end_node} (비용: {cost_type})")
        
        try:
            # 최단 경로 계산
            if cost_type != 'distance':
                # 가중치 기반 최단 경로
                self.build_networkx_graph(cost_type=cost_type)
            
            path = nx.shortest_path(self.graph, start_node, end_node, weight='weight')
            path_length = nx.shortest_path_length(self.graph, start_node, end_node, weight='weight')
            
            # 경로 통계 계산
            path_stats = self._calculate_path_statistics(path)
            
            result = {
                'path': path,
                'total_cost': path_length,
                'cost_type': cost_type,
                'path_length': len(path),
                'total_distance_m': path_stats['total_distance'],
                'total_time_s': path_stats['total_time'],
                'avg_difficulty': path_stats['avg_difficulty'],
                'total_height_change': path_stats['total_height_change']
            }
            
            print(f"✅ 경로 탐색 완료:")
            print(f"   경로 길이: {len(path)}개 노드")
            print(f"   총 거리: {path_stats['total_distance']:.1f}m")
            print(f"   예상 시간: {path_stats['total_time']:.1f}초")
            print(f"   평균 난이도: {path_stats['avg_difficulty']:.3f}")
            
            return result
            
        except nx.NetworkXNoPath:
            print(f"❌ 경로를 찾을 수 없습니다: {start_node} → {end_node}")
            return None
    
    def _calculate_path_statistics(self, path: List[int]) -> Dict:
        """경로 통계 계산"""
        if len(path) < 2:
            return {
                'total_distance': 0,
                'total_time': 0,
                'avg_difficulty': 0,
                'avg_accessibility': 0,
                'total_height_change': 0
            }
        
        total_distance = 0
        total_time = 0
        difficulties = []
        accessibilities = []
        total_height_change = 0
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            if self.graph.has_edge(from_node, to_node):
                edge_data = self.graph[from_node][to_node]
                total_distance += edge_data['distance']
                total_time += edge_data['travel_time']
                total_height_change += edge_data['height_change']
            
            # 노드 정보
            node_data = self.graph.nodes[from_node]
            difficulties.append(node_data['difficulty'])
        
        # 마지막 노드 정보 추가
        last_node_data = self.graph.nodes[path[-1]]
        difficulties.append(last_node_data['difficulty'])
        
        return {
            'total_distance': total_distance,
            'total_time': total_time,
            'avg_difficulty': np.mean(difficulties),
            'total_height_change': total_height_change
        }
    
    def compare_paths(self, start_node: int, end_node: int) -> Dict:
        """다양한 비용 기준으로 경로 비교"""
        cost_types = ['distance', 'difficulty', 'time']
        results = {}
        
        print(f"🔍 다양한 기준으로 경로 비교: {start_node} → {end_node}")
        
        for cost_type in cost_types:
            result = self.find_optimal_path(start_node, end_node, cost_type)
            if result:
                results[cost_type] = result
        
        return results
    
    def visualize_graph(self, save_path: str = None):
        """그래프 시각화"""
        if self.graph is None:
            raise ValueError("그래프를 먼저 생성하세요")
        
        print("📊 그래프 시각화 생성 중...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 위치 정보
        pos = {node: (data['lng'], data['lat']) for node, data in self.graph.nodes(data=True)}
        
        # 1. 난이도별 노드 색상
        ax1 = axes[0, 0]
        node_colors = [self.graph.nodes[node]['difficulty'] for node in self.graph.nodes()]
        nx.draw(self.graph, pos, ax=ax1, node_color=node_colors, cmap='Reds',
                node_size=30, with_labels=False, alpha=0.7)
        ax1.set_title('Nodes by Difficulty')
        
        # 2. 클러스터별 노드 색상
        ax2 = axes[0, 1]
        cluster_colors = [self.graph.nodes[node]['cluster_id'] for node in self.graph.nodes()]
        nx.draw(self.graph, pos, ax=ax2, node_color=cluster_colors, cmap='tab10',
                node_size=30, with_labels=False, alpha=0.7)
        ax2.set_title('Nodes by Cluster')
        
        # 3. 고도별 노드 색상
        ax3 = axes[1, 0]
        height_colors = [self.graph.nodes[node]['height'] for node in self.graph.nodes()]
        nx.draw(self.graph, pos, ax=ax3, node_color=height_colors, cmap='terrain',
                node_size=30, with_labels=False, alpha=0.7)
        ax3.set_title('Nodes by Height')
        
        # 4. 연결성 통계
        ax4 = axes[1, 1]
        degrees = [self.graph.degree(node) for node in self.graph.nodes()]
        ax4.hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Node Degree')
        ax4.set_ylabel('Count')
        ax4.set_title('Node Degree Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 그래프 시각화 저장: {save_path}")
        
        plt.show()
    
    def create_interactive_map(self, save_path: str = None) -> folium.Map:
        """인터랙티브 지도 생성"""
        if not self.nodes:
            raise ValueError("노드를 먼저 생성하세요")
        
        print("🗺️  인터랙티브 지도 생성 중...")
        
        # 중심점 계산
        center_lat = np.mean([node.latitude for node in self.nodes])
        center_lng = np.mean([node.longitude for node in self.nodes])
        
        # 지도 생성
        m = folium.Map(location=[center_lat, center_lng], zoom_start=16)
        
        # 노드 추가
        for node in self.nodes:
            # 난이도에 따른 색상
            if node.difficulty < 0.2:
                color = 'green'
            elif node.difficulty < 0.4:
                color = 'lightgreen'
            elif node.difficulty < 0.6:
                color = 'yellow'
            elif node.difficulty < 0.8:
                color = 'orange'
            else:
                color = 'red'
            
            popup_text = f"""
            Node ID: {node.id}<br>
            Cluster ID: {node.cluster_id}<br>
            Difficulty: {node.difficulty:.3f}<br>
            Height: {node.height:.1f}m<br>
            Velocity: {node.velocity:.1f}m/s
            """
            
            folium.CircleMarker(
                location=[node.latitude, node.longitude],
                radius=5,
                color=color,
                fillColor=color,
                popup=popup_text,
                fillOpacity=0.7
            ).add_to(m)
        
        # 엣지 추가 (주요 연결만)
        if self.graph:
            for edge in list(self.graph.edges())[:100]:  # 처음 100개만 표시
                node1 = self.graph.nodes[edge[0]]
                node2 = self.graph.nodes[edge[1]]
                
                folium.PolyLine(
                    locations=[[node1['lat'], node1['lng']], [node2['lat'], node2['lng']]],
                    color='blue',
                    weight=1,
                    opacity=0.3
                ).add_to(m)
        
        if save_path:
            m.save(save_path)
            print(f"💾 인터랙티브 지도 저장: {save_path}")
        
        return m
    
    def save_graph_data(self, nodes_file: str, edges_file: str):
        """그래프 데이터 저장"""
        # 노드 데이터 저장
        nodes_data = []
        for node in self.nodes:
            nodes_data.append({
                'node_id': node.id,
                'latitude': node.latitude,
                'longitude': node.longitude,
                'height': node.height,
                'difficulty': node.difficulty,
                'velocity': node.velocity,
                'window_id': node.window_id,
                'cluster_id': node.cluster_id
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(nodes_file, index=False)
        print(f"💾 노드 데이터 저장: {nodes_file}")
        
        # 엣지 데이터 저장
        edges_data = []
        for edge in self.edges:
            edges_data.append({
                'from_node': edge.from_node,
                'to_node': edge.to_node,
                'distance_m': edge.distance_m,
                'travel_time_s': edge.travel_time_s,
                'difficulty_cost': edge.difficulty_cost,
                'height_change': edge.height_change
            })
        
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(edges_file, index=False)
        print(f"💾 엣지 데이터 저장: {edges_file}")

def main():
    """메인 실행 함수"""
    print("🗺️  GPS 그래프 빌더 실행")
    print("=" * 60)
    
    try:
        # 그래프 빌더 초기화
        builder = GPSGraphBuilder()
        
        # 1. 향상된 GPS 데이터 로드 (클러스터링 기반 난이도)
        data = builder.load_synchronized_data("results/gps_synchronized.csv")
        
        # 2. 노드 생성
        nodes = builder.create_nodes()
        
        # 3. 엣지 생성 (50m 이내 노드들만 연결)
        edges = builder.create_edges(max_distance_m=50.0)
        
        # 4. NetworkX 그래프 생성
        graph = builder.build_networkx_graph(cost_type='difficulty')
        
        # 5. 샘플 경로 탐색
        if len(nodes) >= 2:
            start_node = nodes[0].id
            end_node = nodes[-1].id
            
            # 다양한 기준으로 경로 비교
            path_results = builder.compare_paths(start_node, end_node)
            
            print(f"\n📊 경로 비교 결과:")
            for cost_type, result in path_results.items():
                if result:
                    print(f"  {cost_type}: {result['total_distance_m']:.1f}m, "
                          f"난이도 {result['avg_difficulty']:.3f}")
        
        # 6. 시각화
        builder.visualize_graph(save_path="results/gps_graph_visualization.png")
        
        # 7. 그래프 데이터 저장
        builder.save_graph_data(
            nodes_file="results/gps_nodes.csv",
            edges_file="results/gps_edges.csv"
        )
        
        print(f"\n✅ GPS 그래프 구축 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()