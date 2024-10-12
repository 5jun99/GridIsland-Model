#!/usr/bin/env python3
"""
휠체어 친화 경로찾기 시스템 - 최종 통합 시스템
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform

# ✅ 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')
else:  # Linux (서버 등)
    rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

import networkx as nx
import folium
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import seaborn as sns

class WheelchairPathfinder:
    """휠체어 사용자를 위한 최적 경로 탐색 시스템"""
    
    def __init__(self):
        self.nodes_df = None
        self.edges_df = None
        self.graph = None
        self.path_results = {}
        
    def load_graph_data(self, nodes_file: str, edges_file: str):
        """그래프 데이터 로드"""
        print(f"📊 그래프 데이터 로드...")
        
        self.nodes_df = pd.read_csv(nodes_file)
        self.edges_df = pd.read_csv(edges_file)
        
        print(f"✅ 로드 완료: {len(self.nodes_df)}개 노드, {len(self.edges_df)}개 엣지")
        
        # NetworkX 그래프 구축
        self._build_networkx_graph()
        
    def _build_networkx_graph(self):
        """NetworkX 그래프 구축"""
        self.graph = nx.Graph()
        
        # 노드 추가
        for _, node in self.nodes_df.iterrows():
            self.graph.add_node(node['node_id'],
                              lat=node['latitude'],
                              lng=node['longitude'],
                              height=node['height'],
                              difficulty=node['difficulty'],
                              cluster_id=node.get('cluster_id', 0))
        
        # 엣지 추가 (중복 제거)
        added_edges = set()
        for _, edge in self.edges_df.iterrows():
            edge_key = tuple(sorted([edge['from_node'], edge['to_node']]))
            
            if edge_key not in added_edges:
                self.graph.add_edge(edge['from_node'], edge['to_node'],
                                   distance=edge['distance_m'],
                                   difficulty_cost=edge['difficulty_cost'],
                                   travel_time=edge['travel_time_s'],
                                   height_change=edge['height_change'])
                added_edges.add(edge_key)
        
        print(f"✅ NetworkX 그래프 구축: {self.graph.number_of_nodes()}개 노드, {self.graph.number_of_edges()}개 엣지")
    
    def find_optimal_path(self, start_node: int, end_node: int) -> Dict:
        """최적 경로 탐색 (난이도 기반)"""
        if self.graph is None:
            raise ValueError("그래프를 먼저 로드하세요")
        
        print(f"🎯 최적 경로 탐색: {start_node} → {end_node}")
        
        try:
            # 난이도 비용 기반 최단 경로
            path = nx.shortest_path(self.graph, start_node, end_node, weight='difficulty_cost')
            total_cost = nx.shortest_path_length(self.graph, start_node, end_node, weight='difficulty_cost')
            
            # 경로 통계 계산
            path_stats = self._calculate_detailed_path_stats(path)
            
            result = {
                'path': path,
                'total_difficulty_cost': total_cost,
                'path_length': len(path),
                'total_distance_m': path_stats['total_distance'],
                'total_time_s': path_stats['total_time'],
                'avg_difficulty': path_stats['avg_difficulty'],
                'total_height_change': path_stats['total_height_change'],
                'difficulty_grade': self._calculate_difficulty_grade(path_stats['avg_difficulty'])
            }
            
            print(f"✅ 최적 경로 발견:")
            print(f"   경로 길이: {len(path)}개 노드")
            print(f"   총 거리: {path_stats['total_distance']:.1f}m")
            print(f"   예상 시간: {path_stats['total_time']/60:.1f}분")
            print(f"   난이도 등급: {result['difficulty_grade']}")
            
            return result
            
        except nx.NetworkXNoPath:
            print(f"❌ 경로를 찾을 수 없습니다: {start_node} → {end_node}")
            return None
    
    def compare_all_path_types(self, start_node: int, end_node: int) -> Dict:
        """모든 경로 유형 비교"""
        cost_types = {
            'distance': '최단거리',
            'difficulty_cost': '난이도 최적',
            'travel_time': '최단시간'
        }
        
        results = {}
        
        print(f"🔍 모든 경로 유형 비교: {start_node} → {end_node}")
        
        for cost_type, description in cost_types.items():
            try:
                path = nx.shortest_path(self.graph, start_node, end_node, weight=cost_type)
                total_cost = nx.shortest_path_length(self.graph, start_node, end_node, weight=cost_type)
                
                path_stats = self._calculate_detailed_path_stats(path)
                
                results[cost_type] = {
                    'path': path,
                    'description': description,
                    'total_cost': total_cost,
                    'path_length': len(path),
                    'total_distance_m': path_stats['total_distance'],
                    'total_time_s': path_stats['total_time'],
                    'avg_difficulty': path_stats['avg_difficulty'],
                    'difficulty_grade': self._calculate_difficulty_grade(path_stats['avg_difficulty'])
                }
                
                print(f"   {description}: 거리 {path_stats['total_distance']:.1f}m, "
                      f"난이도 {path_stats['avg_difficulty']:.3f}")
                
            except nx.NetworkXNoPath:
                print(f"   {description}: 경로 없음")
                results[cost_type] = None
        
        self.path_results = results
        return results
    
    def _calculate_detailed_path_stats(self, path: List[int]) -> Dict:
        """상세 경로 통계 계산"""
        if len(path) < 2:
            return {
                'total_distance': 0, 'total_time': 0,
                'avg_difficulty': 0, 'total_height_change': 0
            }
        
        total_distance = 0
        total_time = 0
        total_height_change = 0
        difficulties = []
        
        # 엣지 기반 통계
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            if self.graph.has_edge(from_node, to_node):
                edge_data = self.graph[from_node][to_node]
                total_distance += edge_data['distance']
                total_time += edge_data['travel_time']
                total_height_change += edge_data['height_change']
        
        # 노드 기반 통계
        for node_id in path:
            node_data = self.graph.nodes[node_id]
            difficulties.append(node_data['difficulty'])
        
        return {
            'total_distance': total_distance,
            'total_time': total_time,
            'avg_difficulty': np.mean(difficulties),
            'total_height_change': total_height_change
        }
    
    def _calculate_difficulty_grade(self, difficulty_score: float) -> str:
        """난이도 등급 계산"""
        if difficulty_score <= 0.2:
            return "A+ (매우 쉬움)"
        elif difficulty_score <= 0.4:
            return "A (쉬움)"
        elif difficulty_score <= 0.6:
            return "B (보통)"
        elif difficulty_score <= 0.8:
            return "C (어려움)"
        else:
            return "D (매우 어려움)"
    
    
    def create_comprehensive_map(self, path_results: Dict = None, save_path: str = None) -> folium.Map:
        """종합 인터랙티브 지도 생성"""
        if path_results is None:
            path_results = self.path_results
        
        print("🗺️  종합 인터랙티브 지도 생성 중...")
        
        # 중심점 계산
        center_lat = self.nodes_df['latitude'].mean()
        center_lng = self.nodes_df['longitude'].mean()
        
        # 지도 생성
        m = folium.Map(location=[center_lat, center_lng], zoom_start=16)
        
        # 모든 노드 추가 (난이도에 따른 색상)
        for _, node in self.nodes_df.iterrows():
            # 난이도에 따른 색상
            difficulty = node['difficulty']
            if difficulty <= 0.2:
                color = 'green'
                grade = 'A+'
            elif difficulty <= 0.4:
                color = 'lightgreen'
                grade = 'A'
            elif difficulty <= 0.6:
                color = 'yellow'
                grade = 'B'
            elif difficulty <= 0.8:
                color = 'orange'
                grade = 'C'
            else:
                color = 'red'
                grade = 'D'
            
            cluster_id = node.get('cluster_id', 'N/A')
            popup_text = f"""
            <b>노드 {node['node_id']}</b><br>
            클러스터: {cluster_id}<br>
            난이도: {difficulty:.3f} ({grade}등급)<br>
            고도: {node['height']:.1f}m<br>
            속도: {node['velocity']:.1f}m/s
            """
            
            folium.CircleMarker(
                location=[node['latitude'], node['longitude']],
                radius=4,
                color='black',
                fillColor=color,
                popup=popup_text,
                fillOpacity=0.7,
                weight=1
            ).add_to(m)
        
        # 경로들 추가
        if path_results:
            colors = {'distance': 'blue', 'difficulty_cost': 'orange', 'travel_time': 'purple'}
            
            for cost_type, result in path_results.items():
                if result and result['path']:
                    path_coords = []
                    for node_id in result['path']:
                        node_data = self.nodes_df[self.nodes_df['node_id'] == node_id].iloc[0]
                        path_coords.append([node_data['latitude'], node_data['longitude']])
                    
                    folium.PolyLine(
                        locations=path_coords,
                        color=colors.get(cost_type, 'gray'),
                        weight=4,
                        opacity=0.8,
                        popup=f"{result['description']}: {result['total_distance_m']:.1f}m"
                    ).add_to(m)
            
            # 범례 추가
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 200px; height: 100px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <p><b>경로 유형</b></p>
            <p><i class="fa fa-minus" style="color:blue"></i> 최단거리</p>
            <p><i class="fa fa-minus" style="color:orange"></i> 난이도 최적</p>
            <p><i class="fa fa-minus" style="color:purple"></i> 최단시간</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
        
        if save_path:
            m.save(save_path)
            print(f"💾 종합 지도 저장: {save_path}")
        
        return m
    
    def create_analysis_dashboard(self, path_results: Dict = None, save_path: str = None):
        """분석 대시보드 생성"""
        if path_results is None:
            path_results = self.path_results
        
        if not path_results:
            print("❌ 분석할 경로 결과가 없습니다")
            return
        
        print("📊 분석 대시보드 생성 중...")
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 데이터 준비
        valid_results = {k: v for k, v in path_results.items() if v is not None}
        
        if not valid_results:
            print("❌ 유효한 경로 결과가 없습니다")
            return
        
        route_names = [v['description'] for v in valid_results.values()]
        distances = [v['total_distance_m'] for v in valid_results.values()]
        times = [v['total_time_s']/60 for v in valid_results.values()]  # 분 단위
        difficulties = [v['avg_difficulty'] for v in valid_results.values()]
        path_lengths = [v['path_length'] for v in valid_results.values()]
        
        # 1. 거리 비교
        ax1 = axes[0, 0]
        bars1 = ax1.bar(route_names, distances, color=['skyblue', 'lightgreen', 'orange', 'purple'])
        ax1.set_title('경로별 총 거리 비교', fontsize=14, fontweight='bold')
        ax1.set_ylabel('거리 (m)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, distance in zip(bars1, distances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{distance:.1f}m', ha='center', va='bottom')
        
        # 2. 시간 비교
        ax2 = axes[0, 1]
        bars2 = ax2.bar(route_names, times, color=['skyblue', 'lightgreen', 'orange', 'purple'])
        ax2.set_title('경로별 예상 시간 비교', fontsize=14, fontweight='bold')
        ax2.set_ylabel('시간 (분)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, time in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time:.1f}분', ha='center', va='bottom')
        
        # 3. 난이도 비교 (상세)
        ax3 = axes[0, 2]
        bars3 = ax3.bar(route_names, difficulties, color=['skyblue', 'lightgreen', 'orange', 'purple'])
        ax3.set_title('경로별 난이도 점수', fontsize=14, fontweight='bold')
        ax3.set_ylabel('난이도 점수 (0-1)')
        ax3.set_ylim(0, max(difficulties) * 1.2 if difficulties else 1)
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, diff in zip(bars3, difficulties):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{diff:.3f}', ha='center', va='bottom')
        
        # 4. 경로 효율성 비교 (거리 대비 시간)
        ax4 = axes[1, 0]
        if distances and times:
            efficiency = [d/t if t > 0 else 0 for d, t in zip(distances, times)]
            bars4 = ax4.bar(route_names, efficiency, color=['skyblue', 'lightgreen', 'orange', 'purple'])
            ax4.set_title('경로별 효율성 (거리/시간)', fontsize=14, fontweight='bold')
            ax4.set_ylabel('효율성 (m/분)')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, eff in zip(bars4, efficiency):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{eff:.1f}', ha='center', va='bottom')
        
        # 5. 경로 길이 비교
        ax5 = axes[1, 1]
        bars5 = ax5.bar(route_names, path_lengths, color=['skyblue', 'lightgreen', 'orange', 'purple'])
        ax5.set_title('경로별 노드 수 비교', fontsize=14, fontweight='bold')
        ax5.set_ylabel('노드 수')
        ax5.tick_params(axis='x', rotation=45)
        
        for bar, length in zip(bars5, path_lengths):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{length}개', ha='center', va='bottom')
        
        # 6. 종합 점수 레이더 차트
        ax6 = axes[1, 2]
        
        if distances and times and difficulties:
            # 정규화된 점수 계산 (높을수록 좋음)
            norm_distances = [1 - (d - min(distances)) / (max(distances) - min(distances)) if max(distances) > min(distances) else 0.5 for d in distances]
            norm_times = [1 - (t - min(times)) / (max(times) - min(times)) if max(times) > min(times) else 0.5 for t in times]
            norm_difficulties = [1 - d for d in difficulties]  # 난이도는 낮을수록 좋음
            
            categories = ['거리', '시간', '안전성', '효율성']
            
            # 난이도 최적 경로만 레이더 차트로 표시
            difficulty_idx = next((i for i, k in enumerate(valid_results.keys()) if k == 'difficulty_cost'), 0)
            
            values = [
                norm_distances[difficulty_idx] if difficulty_idx < len(norm_distances) else 0.5,
                norm_times[difficulty_idx] if difficulty_idx < len(norm_times) else 0.5,
                norm_difficulties[difficulty_idx] if difficulty_idx < len(norm_difficulties) else 0.5,
                efficiency[difficulty_idx] / max(efficiency) if difficulty_idx < len(efficiency) and max(efficiency) > 0 else 0.5
            ]
        
        # 레이더 차트
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 첫 번째 값을 마지막에 추가하여 닫힌 도형 만들기
        angles += angles[:1]
        
        ax6.plot(angles, values, 'o-', linewidth=2, color='green')
        ax6.fill(angles, values, alpha=0.25, color='green')
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 1)
        ax6.set_title('난이도 최적 경로 종합 평가', fontsize=14, fontweight='bold')
        ax6.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 분석 대시보드 저장: {save_path}")
        
        plt.show()
    
    def generate_recommendation_report(self, path_results: Dict = None) -> str:
        """추천 보고서 생성"""
        if path_results is None:
            path_results = self.path_results
        
        if not path_results:
            return "❌ 분석할 경로 결과가 없습니다"
        
        report = []
        report.append("♿ 휠체어 사용자 경로 추천 보고서")
        report.append("=" * 60)
        
        # 난이도 최적 경로 분석
        optimal_result = path_results.get('difficulty_cost')
        if optimal_result:
            report.append(f"\n🎯 추천 경로: 난이도 최적화 경로")
            report.append(f"   총 거리: {optimal_result['total_distance_m']:.1f}m")
            report.append(f"   예상 시간: {optimal_result['total_time_s']/60:.1f}분")
            report.append(f"   난이도 등급: {optimal_result['difficulty_grade']}")
            
            # 클러스터 기반 분석 추가
            path_clusters = []
            for node_id in optimal_result['path']:
                node_data = self.graph.nodes[node_id]
                cluster_id = node_data.get('cluster_id', 0)
                path_clusters.append(cluster_id)
            
            cluster_distribution = {}
            for cluster in path_clusters:
                cluster_distribution[cluster] = cluster_distribution.get(cluster, 0) + 1
            
            report.append(f"\n📊 클러스터 기반 경로 분석:")
            for cluster_id, count in sorted(cluster_distribution.items()):
                percentage = (count / len(path_clusters)) * 100
                report.append(f"   클러스터 {cluster_id}: {count}개 노드 ({percentage:.1f}%)")
            
            # 권장사항
            difficulty = optimal_result['avg_difficulty']
            if difficulty <= 0.2:
                report.append("✅ 이 경로는 휠체어 이용에 매우 적합합니다.")
            elif difficulty <= 0.4:
                report.append("✅ 이 경로는 휠체어 이용에 적합합니다.")
            elif difficulty <= 0.6:
                report.append("⚠️  이 경로는 휠체어 이용이 가능하지만 주의가 필요합니다.")
            else:
                report.append("❌ 이 경로는 휠체어 이용이 어려울 수 있습니다.")
        
        # 다른 경로들과 비교
        report.append(f"\n📊 다른 경로 옵션과의 비교:")
        
        for cost_type, result in path_results.items():
            if result and cost_type != 'difficulty_cost':
                diff_distance = result['total_distance_m'] - optimal_result['total_distance_m']
                diff_time = (result['total_time_s'] - optimal_result['total_time_s']) / 60
                
                if diff_distance > 0:
                    distance_text = f"+{diff_distance:.1f}m 더 긺"
                else:
                    distance_text = f"{abs(diff_distance):.1f}m 더 짧음"
                
                if diff_time > 0:
                    time_text = f"+{diff_time:.1f}분 더 소요"
                else:
                    time_text = f"{abs(diff_time):.1f}분 단축"
                
                report.append(f"   {result['description']}: {distance_text}, {time_text}")
                report.append(f"     난이도: {result['avg_difficulty']:.3f}")
        
        # 주의사항
        report.append(f"\n⚠️  주의사항:")
        report.append("   - 실제 이동 시에는 노면 상태, 날씨 등을 추가로 고려하세요")
        report.append("   - 급한 경사나 장애물이 있을 수 있으니 현장 확인이 필요합니다")
        report.append("   - 휠체어 종류와 사용자 능력에 따라 적합성이 달라질 수 있습니다")
        
        return "\n".join(report)

def main():
    """메인 실행 함수"""
    print("♿ 휠체어 친화 경로찾기 시스템")
    print("=" * 60)
    
    try:
        # 경로찾기 시스템 초기화
        pathfinder = WheelchairPathfinder()
        
        # 1. 그래프 데이터 로드
        pathfinder.load_graph_data(
            nodes_file="results/gps_nodes.csv",
            edges_file="results/gps_edges.csv"
        )
        
        # 2. 샘플 경로 분석
        start_node = 0
        end_node = pathfinder.nodes_df['node_id'].max()
        
        print(f"\n🎯 샘플 경로 분석: 노드 {start_node} → 노드 {end_node}")
        
        # 3. 모든 경로 유형 비교
        path_results = pathfinder.compare_all_path_types(start_node, end_node)
        
        # 4. 최적 경로 상세 분석
        optimal_result = pathfinder.find_optimal_path(start_node, end_node)
        
        # 5. 종합 지도 생성
        comprehensive_map = pathfinder.create_comprehensive_map(
            path_results, 
            save_path="results/wheelchair_comprehensive_map.html"
        )
        
        # 6. 분석 대시보드 생성
        pathfinder.create_analysis_dashboard(
            path_results,
            save_path="results/wheelchair_analysis_dashboard.png"
        )
        
        # 7. 추천 보고서 생성
        report = pathfinder.generate_recommendation_report(path_results)
        print(f"\n{report}")
        
        # 보고서 저장
        with open("results/wheelchair_recommendation_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\n✅ 휠체어 친화 경로찾기 시스템 완료!")
        print(f"📁 생성된 파일들:")
        print(f"   - results/wheelchair_comprehensive_map.html (종합 지도)")
        print(f"   - results/wheelchair_analysis_dashboard.png (분석 대시보드)")
        print(f"   - results/wheelchair_recommendation_report.txt (추천 보고서)")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()