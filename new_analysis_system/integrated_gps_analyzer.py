#!/usr/bin/env python3
"""
통합 GPS 분석기 - 모든 경로 데이터를 통합하여 분석하고 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx
from gps_loader import EnhancedGPSLoader
from gps_graph_builder import GPSGraphBuilder
import seaborn as sns

class IntegratedGPSAnalyzer:
    """실제 데이터와 시뮬레이션 데이터를 통합 분석하는 시스템"""
    
    def __init__(self):
        self.all_routes_data = {}
        self.combined_graph = None
        self.path_results = {}
        
    def load_all_routes(self, route_summary_file="data/simulated_routes/route_summary.csv"):
        """모든 경로 데이터 로드 및 분석"""
        print("🚀 통합 GPS 분석 시작")
        print("=" * 60)
        
        # 경로 요약 파일 로드
        summary_df = pd.read_csv(route_summary_file)
        
        print(f"📊 총 {len(summary_df)}개 경로 통합 분석")
        
        for _, route in summary_df.iterrows():
            route_id = route['route_id']
            route_path = route['data_path']
            route_name = route['name']
            
            print(f"\n🛣️  {route_name} 분석 중...")
            
            try:
                # GPS 로더로 각 경로 분석
                loader = EnhancedGPSLoader(route_path)
                
                # 전체 분석 파이프라인 실행
                gps_data = loader.load_gps_data()
                sensor_data = loader.load_sensor_data()
                synchronized_data = loader.synchronize_data(window_size=1.0)
                features = loader.extract_advanced_features(window_size=150, overlap_ratio=0.6)
                clustered_data = loader.perform_clustering(n_clusters=4)
                difficulty_results = loader.analyze_difficulty()
                gps_with_difficulty = loader.map_difficulty_to_gps()
                
                # 결과 저장
                self.all_routes_data[route_id] = {
                    'info': route.to_dict(),
                    'gps_data': gps_data,
                    'gps_with_difficulty': gps_with_difficulty,
                    'difficulty_results': difficulty_results,
                    'features': features,
                    'clustered_data': clustered_data,
                    'avg_difficulty': gps_with_difficulty['difficulty'].mean(),
                    'difficulty_std': gps_with_difficulty['difficulty'].std()
                }
                
                print(f"   ✅ 완료: 평균 난이도 {gps_with_difficulty['difficulty'].mean():.3f}")
                
            except Exception as e:
                print(f"   ❌ 실패: {e}")
                continue
        
        print(f"\n✅ 통합 분석 완료: {len(self.all_routes_data)}개 경로")
        return self.all_routes_data
    
    def build_integrated_graph(self):
        """모든 경로 데이터를 통합한 단일 그래프 구축"""
        print("\n🔗 통합 그래프 구축 중...")
        
        all_nodes = []
        all_edges = []
        node_id_counter = 0
        
        # 각 경로의 GPS 포인트를 노드로 변환
        for route_id, route_data in self.all_routes_data.items():
            gps_with_difficulty = route_data['gps_with_difficulty']
            route_info = route_data['info']
            
            # 각 GPS 포인트를 노드로 추가
            for idx, row in gps_with_difficulty.iterrows():
                node = {
                    'node_id': node_id_counter,
                    'route_id': route_id,
                    'route_name': route_info['name'],
                    'route_type': route_info['difficulty_type'],
                    'latitude': row['lat_mean'],
                    'longitude': row['lng_mean'],
                    'height': row['height_mean'],
                    'difficulty': row['difficulty'],
                    'cluster_id': row.get('cluster_id', 0),
                    'window_id': idx
                }
                all_nodes.append(node)
                node_id_counter += 1
        
        # 동일 경로 내 연속 노드들을 엣지로 연결
        for route_id, route_data in self.all_routes_data.items():
            route_nodes = [n for n in all_nodes if n['route_id'] == route_id]
            
            for i in range(len(route_nodes) - 1):
                from_node = route_nodes[i]
                to_node = route_nodes[i + 1]
                
                # 거리 계산
                distance = self._calculate_distance(
                    from_node['latitude'], from_node['longitude'],
                    to_node['latitude'], to_node['longitude']
                )
                
                # 평균 난이도
                avg_difficulty = (from_node['difficulty'] + to_node['difficulty']) / 2
                
                edge = {
                    'from_node': from_node['node_id'],
                    'to_node': to_node['node_id'],
                    'route_id': route_id,
                    'distance': distance,
                    'difficulty': avg_difficulty,
                    'travel_time': distance / 1.5,  # 1.5 m/s 가정
                }
                all_edges.append(edge)
        
        # DataFrame으로 변환
        self.nodes_df = pd.DataFrame(all_nodes)
        self.edges_df = pd.DataFrame(all_edges)
        
        # NetworkX 그래프 구축
        self.combined_graph = nx.DiGraph()
        
        # 노드 추가
        for _, node in self.nodes_df.iterrows():
            self.combined_graph.add_node(
                node['node_id'],
                pos=(node['longitude'], node['latitude']),
                difficulty=node['difficulty'],
                route_id=node['route_id'],
                route_type=node['route_type']
            )
        
        # 엣지 추가
        for _, edge in self.edges_df.iterrows():
            self.combined_graph.add_edge(
                edge['from_node'], edge['to_node'],
                distance=edge['distance'],
                difficulty=edge['difficulty'],
                travel_time=edge['travel_time']
            )
        
        print(f"✅ 통합 그래프 완성: {len(all_nodes)}개 노드, {len(all_edges)}개 엣지")
        return self.combined_graph
    
    def find_optimal_paths(self, start_route='real_original', end_route='flat_bypass'):
        """여러 기준으로 최적 경로 탐색"""
        print(f"\n🎯 최적 경로 탐색: {start_route} → {end_route}")
        
        # 시작점과 끝점 노드 찾기
        start_nodes = self.nodes_df[self.nodes_df['route_id'] == start_route]['node_id'].tolist()
        end_nodes = self.nodes_df[self.nodes_df['route_id'] == end_route]['node_id'].tolist()
        
        if not start_nodes or not end_nodes:
            print("❌ 시작점 또는 끝점을 찾을 수 없습니다")
            return {}
        
        start_node = start_nodes[0]  # 첫 번째 노드
        end_node = end_nodes[-1]     # 마지막 노드
        
        results = {}
        
        try:
            # 1. 최단거리 경로
            shortest_path = nx.shortest_path(
                self.combined_graph, start_node, end_node, weight='distance'
            )
            shortest_distance = nx.shortest_path_length(
                self.combined_graph, start_node, end_node, weight='distance'
            )
            shortest_difficulty = self._calculate_path_difficulty(shortest_path)
            
            results['shortest_distance'] = {
                'path': shortest_path,
                'distance': shortest_distance,
                'difficulty': shortest_difficulty,
                'type': '최단거리'
            }
            
            # 2. 최소난이도 경로
            min_difficulty_path = nx.shortest_path(
                self.combined_graph, start_node, end_node, weight='difficulty'
            )
            min_difficulty_distance = self._calculate_path_distance(min_difficulty_path)
            min_difficulty_difficulty = nx.shortest_path_length(
                self.combined_graph, start_node, end_node, weight='difficulty'
            )
            
            results['min_difficulty'] = {
                'path': min_difficulty_path,
                'distance': min_difficulty_distance,
                'difficulty': min_difficulty_difficulty,
                'type': '최소난이도'
            }
            
            # 3. 최단시간 경로
            fastest_path = nx.shortest_path(
                self.combined_graph, start_node, end_node, weight='travel_time'
            )
            fastest_distance = self._calculate_path_distance(fastest_path)
            fastest_difficulty = self._calculate_path_difficulty(fastest_path)
            
            results['fastest'] = {
                'path': fastest_path,
                'distance': fastest_distance,
                'difficulty': fastest_difficulty,
                'type': '최단시간'
            }
            
            print(f"✅ 경로 탐색 완료:")
            for key, result in results.items():
                print(f"   {result['type']}: 거리 {result['distance']:.1f}m, 난이도 {result['difficulty']:.3f}")
            
        except nx.NetworkXNoPath:
            print("❌ 경로를 찾을 수 없습니다")
        
        self.path_results = results
        return results
    
    def create_comprehensive_map(self, save_path="results/integrated_gps_map.html"):
        """모든 경로와 최적 경로를 표시하는 종합 지도 생성"""
        print(f"\n🗺️  종합 지도 생성 중...")
        
        # 중심점 계산
        center_lat = self.nodes_df['latitude'].mean()
        center_lon = self.nodes_df['longitude'].mean()
        
        # Folium 지도 생성
        m = folium.Map(location=[center_lat, center_lon], zoom_start=16)
        
        # 경로별 색상 정의
        route_colors = {
            'real_original': 'red',
            'real_measured': 'darkred', 
            'flat_bypass': 'green',
            'slope_route': 'orange',
            'stairs_shortcut': 'purple',
            'rough_path': 'brown',
            'mixed_complex': 'pink'
        }
        
        # 각 경로의 GPS 트랙 표시
        for route_id, route_data in self.all_routes_data.items():
            gps_data = route_data['gps_with_difficulty']
            route_info = route_data['info']
            color = route_colors.get(route_id, 'gray')
            
            # GPS 트랙 라인
            coordinates = [[row['lat_mean'], row['lng_mean']] for _, row in gps_data.iterrows()]
            
            folium.PolyLine(
                coordinates,
                color=color,
                weight=4,
                opacity=0.8,
                popup=f"{route_info['name']}<br>평균 난이도: {route_data['avg_difficulty']:.3f}"
            ).add_to(m)
            
            # 시작점과 끝점 마커
            if len(coordinates) > 0:
                folium.Marker(
                    coordinates[0],
                    popup=f"🚀 {route_info['name']} 시작",
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(m)
                
                folium.Marker(
                    coordinates[-1],
                    popup=f"🏁 {route_info['name']} 끝",
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(m)
        
        # 최적 경로들 표시 (있는 경우)
        if self.path_results:
            for path_type, path_info in self.path_results.items():
                path_nodes = path_info['path']
                
                # 경로 좌표 추출
                path_coords = []
                for node_id in path_nodes:
                    node_data = self.nodes_df[self.nodes_df['node_id'] == node_id].iloc[0]
                    path_coords.append([node_data['latitude'], node_data['longitude']])
                
                # 경로 타입별 스타일
                if path_type == 'shortest_distance':
                    line_color = 'blue'
                    line_weight = 6
                    line_dash = '5, 5'
                elif path_type == 'min_difficulty':
                    line_color = 'cyan'
                    line_weight = 6
                    line_dash = '10, 5'
                else:  # fastest
                    line_color = 'yellow'
                    line_weight = 6
                    line_dash = '15, 5'
                
                folium.PolyLine(
                    path_coords,
                    color=line_color,
                    weight=line_weight,
                    opacity=0.9,
                    dash_array=line_dash,
                    popup=f"{path_info['type']}<br>거리: {path_info['distance']:.1f}m<br>난이도: {path_info['difficulty']:.3f}"
                ).add_to(m)
        
        # 범례 추가
        legend_html = '''
        <div style="position: fixed; 
                   bottom: 50px; left: 50px; width: 300px; height: 200px; 
                   background-color: white; border:2px solid grey; z-index:9999; 
                   font-size:14px; padding: 10px">
        <h4>경로 범례</h4>
        <p><i style="color:red">■</i> 실제 측정 원본</p>
        <p><i style="color:green">■</i> 평지 우회로</p>
        <p><i style="color:orange">■</i> 언덕길</p>
        <p><i style="color:purple">■</i> 계단 지름길</p>
        <p><i style="color:brown">■</i> 울퉁불퉁한 길</p>
        <p><i style="color:blue">---</i> 최단거리</p>
        <p><i style="color:cyan">---</i> 최소난이도</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # 지도 저장
        m.save(save_path)
        print(f"💾 종합 지도 저장: {save_path}")
        
        return m
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """두 GPS 좌표 간 거리 계산 (미터)"""
        from math import radians, cos, sin, asin, sqrt
        
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
            edge_data = self.combined_graph[path[i]][path[i+1]]
            total_distance += edge_data['distance']
        return total_distance
    
    def _calculate_path_difficulty(self, path):
        """경로의 평균 난이도 계산"""
        difficulties = []
        for node_id in path:
            node_data = self.nodes_df[self.nodes_df['node_id'] == node_id].iloc[0]
            difficulties.append(node_data['difficulty'])
        return np.mean(difficulties)
    
    def generate_comparison_report(self):
        """경로 비교 보고서 생성"""
        print("\n📊 통합 분석 보고서 생성")
        
        report = []
        report.append("🚀 통합 GPS 경로 분석 보고서")
        report.append("=" * 60)
        report.append(f"📅 분석 경로 수: {len(self.all_routes_data)}개")
        report.append("")
        
        # 경로별 요약
        report.append("📊 경로별 분석 결과:")
        for route_id, route_data in self.all_routes_data.items():
            route_info = route_data['info']
            report.append(f"\n🛣️  {route_info['name']}")
            report.append(f"   ├─ 유형: {route_info['difficulty_type']}")
            report.append(f"   ├─ 거리: {route_info['distance_m']}m")
            report.append(f"   ├─ 평균 난이도: {route_data['avg_difficulty']:.3f}")
            report.append(f"   └─ 난이도 편차: {route_data['difficulty_std']:.3f}")
        
        # 최적 경로 결과
        if self.path_results:
            report.append("\n🎯 최적 경로 탐색 결과:")
            for path_type, path_info in self.path_results.items():
                report.append(f"\n   {path_info['type']}:")
                report.append(f"   ├─ 총 거리: {path_info['distance']:.1f}m")
                report.append(f"   ├─ 평균 난이도: {path_info['difficulty']:.3f}")
                report.append(f"   └─ 경유 노드: {len(path_info['path'])}개")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # 파일로 저장
        with open("results/integrated_analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n💾 보고서 저장: results/integrated_analysis_report.txt")
        
        return report_text

def main():
    """메인 실행 함수"""
    analyzer = IntegratedGPSAnalyzer()
    
    # 1. 모든 경로 로드 및 분석
    analyzer.load_all_routes()
    
    # 2. 통합 그래프 구축
    analyzer.build_integrated_graph()
    
    # 3. 최적 경로 탐색
    analyzer.find_optimal_paths()
    
    # 4. 종합 지도 생성
    analyzer.create_comprehensive_map()
    
    # 5. 비교 보고서 생성
    analyzer.generate_comparison_report()
    
    print("\n✅ 통합 GPS 분석 완료!")

if __name__ == "__main__":
    main()