#!/usr/bin/env python3
"""
Grid Island 최종 시스템: IMU → 그래프 경로 최적화
완전한 원클릭 솔루션
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import heapq
from final_improved_model import FinalImprovedModel
from utils.data_loader import load_sensor_data, combine_sensor_data

class GridIslandSystem:
    """Grid Island 완전 통합 시스템"""

    def __init__(self):
        self.pipeline = FinalImprovedModel()
        self.nodes_df = None
        self.edges_df = None
        self.adjacency_list = None
        self.is_trained = False

    def setup_system(self):
        """시스템 초기 설정 (모델 학습 포함)"""
        print("🚀 Grid Island 시스템 초기화")
        print("="*50)

        # 모델이 이미 학습되어 있는지 확인
        model_path = 'models/production_pipeline.pkl'
        if os.path.exists(model_path):
            try:
                self.pipeline.load_artifacts(model_path)
                self.is_trained = True
                print("✅ 기존 학습된 모델 로드 완료")
            except:
                print("⚠️  기존 모델 로드 실패, 새로 학습합니다...")
                self.train_model()
        else:
            print("📚 모델 학습 시작...")
            self.train_model()

    def train_model(self):
        """모델 학습"""
        performance = self.pipeline.train_optimized_model()
        self.is_trained = True
        print(f"✅ 모델 학습 완료 (F1: {performance:.4f})")

    def process_imu_data(self, data_dir="test 2025-09-22 18-30-21"):
        """IMU 데이터 처리하여 그래프 생성"""
        if not self.is_trained:
            raise ValueError("시스템이 초기화되지 않았습니다. setup_system()을 먼저 실행하세요.")

        print(f"📊 IMU 데이터 처리: {data_dir}")

        # 센서 데이터 로드
        sensor_data = load_sensor_data(data_dir)
        combined_df = combine_sensor_data(sensor_data)

        # 온라인 추론
        results = self.pipeline.online_infer(combined_df)

        if results is None:
            print("❌ 추론 실패")
            return False

        # 그래프 데이터 생성
        self._create_graph_from_results(results)

        print(f"✅ 그래프 생성: {len(self.nodes_df)}개 노드, {len(self.edges_df)}개 엣지")
        return True

    def _create_graph_from_results(self, results):
        """결과에서 그래프 데이터 생성"""
        positions = results['positions']
        difficulty_smoothed = results['difficulty_smoothed']
        edge_costs = results['edge_costs']
        probabilities = results['probabilities']

        # 노드 생성
        nodes = []
        for i, (start, end) in enumerate(positions):
            node = {
                'node_id': i,
                'start_sample': start,
                'end_sample': end,
                'center_sample': (start + end) // 2,
                'difficulty': int(difficulty_smoothed[i]),
                'difficulty_name': self.pipeline.difficulty_map[difficulty_smoothed[i]],
                'confidence': float(probabilities[i].max()),
                'base_cost': float(edge_costs[i])
            }
            nodes.append(node)

        # 엣지 생성 (다양한 연결 패턴)
        edges = []
        edge_id = 0

        # 1. 순차적 연결 (기본 경로)
        for i in range(len(nodes) - 1):
            current = nodes[i]
            next_node = nodes[i + 1]

            edge = {
                'edge_id': edge_id,
                'from_node': current['node_id'],
                'to_node': next_node['node_id'],
                'cost': (current['base_cost'] + next_node['base_cost']) / 2,
                'distance': abs(next_node['center_sample'] - current['center_sample']),
                'difficulty_avg': (current['difficulty'] + next_node['difficulty']) / 2,
                'edge_type': 'sequential'
            }
            edges.append(edge)
            edge_id += 1

        # 2. 건너뛰기 연결 (우회 경로)
        for jump in [2, 3, 5]:  # 2, 3, 5 노드 건너뛰기
            for i in range(len(nodes) - jump):
                current = nodes[i]
                target = nodes[i + jump]

                # 건너뛰기 페널티 적용
                jump_penalty = 1.0 + (jump - 1) * 0.1  # 멀수록 약간의 페널티

                edge = {
                    'edge_id': edge_id,
                    'from_node': current['node_id'],
                    'to_node': target['node_id'],
                    'cost': (current['base_cost'] + target['base_cost']) / 2 * jump_penalty,
                    'distance': abs(target['center_sample'] - current['center_sample']),
                    'difficulty_avg': (current['difficulty'] + target['difficulty']) / 2,
                    'edge_type': f'jump_{jump}'
                }
                edges.append(edge)
                edge_id += 1

        self.nodes_df = pd.DataFrame(nodes)
        self.edges_df = pd.DataFrame(edges)

        # 인접 리스트 생성
        self._build_adjacency_list()

    def _build_adjacency_list(self):
        """그래프의 인접 리스트 구성"""
        self.adjacency_list = {}

        for _, edge in self.edges_df.iterrows():
            from_node = edge['from_node']
            to_node = edge['to_node']
            cost = edge['cost']

            if from_node not in self.adjacency_list:
                self.adjacency_list[from_node] = []

            self.adjacency_list[from_node].append((to_node, cost))

    def find_optimal_path(self, start_node=0, end_node=None, preference='balanced'):
        """최적 경로 탐색 (다익스트라)"""
        if self.adjacency_list is None:
            print("❌ 그래프가 생성되지 않았습니다.")
            return None

        if end_node is None:
            end_node = len(self.nodes_df) - 1

        print(f"🗺️  경로 탐색: 노드 {start_node} → 노드 {end_node} ({preference})")

        # 선호도별 가중치
        preferences = {
            'fastest': {'distance_weight': 1.0, 'difficulty_weight': 0.3},
            'balanced': {'distance_weight': 1.0, 'difficulty_weight': 1.0},
            'safest': {'distance_weight': 0.7, 'difficulty_weight': 2.0}
        }

        weights = preferences.get(preference, preferences['balanced'])

        # 다익스트라 알고리즘
        distances = {node: float('inf') for node in range(len(self.nodes_df))}
        distances[start_node] = 0
        previous = {}
        visited = set()

        pq = [(0, start_node)]

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == end_node:
                break

            if current_node in self.adjacency_list:
                for neighbor, base_cost in self.adjacency_list[current_node]:
                    if neighbor in visited:
                        continue

                    # 선호도 반영한 비용
                    neighbor_difficulty = self.nodes_df.iloc[neighbor]['difficulty']
                    adjusted_cost = (base_cost * weights['distance_weight'] +
                                   neighbor_difficulty * 20 * weights['difficulty_weight'])

                    new_dist = current_dist + adjusted_cost

                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current_node
                        heapq.heappush(pq, (new_dist, neighbor))

        # 경로 재구성
        if end_node not in previous and start_node != end_node:
            return None, None

        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = previous.get(current)

        path.reverse()

        # 경로 분석
        path_info = self._analyze_path(path, distances[end_node])

        return path, path_info

    def _analyze_path(self, path, total_cost):
        """경로 분석"""
        if len(path) < 2:
            return {'total_cost': total_cost, 'segments': 0, 'difficulties': {}}

        path_nodes = self.nodes_df.iloc[path]

        difficulty_counts = path_nodes['difficulty'].value_counts().to_dict()
        total_distance = path_nodes.iloc[-1]['center_sample'] - path_nodes.iloc[0]['center_sample']

        path_info = {
            'total_cost': total_cost,
            'total_distance': total_distance,
            'segments': len(path),
            'avg_confidence': path_nodes['confidence'].mean(),
            'difficulty_distribution': {
                self.pipeline.difficulty_map[k]: v
                for k, v in difficulty_counts.items()
            }
        }

        return path_info

    def run_complete_analysis(self, data_dir="test 2025-09-22 18-30-21"):
        """완전한 분석 실행"""
        print("🎯 Grid Island 완전 분석 시스템")
        print("="*60)

        # 1. 시스템 초기화
        if not self.is_trained:
            self.setup_system()

        # 2. IMU 데이터 처리
        success = self.process_imu_data(data_dir)
        if not success:
            return False

        # 3. 경로 비교
        preferences = ['fastest', 'balanced', 'safest']
        results = {}

        print(f"\n🔍 경로 옵션 분석")
        print("-"*40)

        for pref in preferences:
            path, info = self.find_optimal_path(preference=pref)
            if path and info:
                results[pref] = {'path': path, 'info': info}

                print(f"\n[{pref.upper()} 경로]")
                print(f"  총 비용: {info['total_cost']:.1f}")
                print(f"  세그먼트: {info['segments']}개")
                print(f"  신뢰도: {info['avg_confidence']:.3f}")

                for difficulty, count in info['difficulty_distribution'].items():
                    print(f"  {difficulty}: {count}개")

        # 4. 데이터 저장
        self.save_results()

        print(f"\n🎉 분석 완료!")
        print(f"📊 그래프: {len(self.nodes_df)}개 노드, {len(self.edges_df)}개 엣지")
        print(f"📁 결과: results/ 폴더 확인")

        return results

    def save_results(self, output_dir="results"):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)

        if self.nodes_df is not None:
            self.nodes_df.to_csv(f"{output_dir}/grid_island_nodes.csv", index=False)

        if self.edges_df is not None:
            self.edges_df.to_csv(f"{output_dir}/grid_island_edges.csv", index=False)

        print(f"💾 결과 저장: {output_dir}/")

def main():
    """메인 실행"""
    system = GridIslandSystem()
    results = system.run_complete_analysis()

    if results:
        print(f"\n✅ Grid Island 시스템 실행 성공!")
        print(f"🎯 {len(results)}가지 경로 옵션 제공")
    else:
        print("❌ 시스템 실행 실패")

if __name__ == "__main__":
    main()