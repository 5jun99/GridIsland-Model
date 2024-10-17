#!/usr/bin/env python3
"""
난이도 분석기 - 클러스터별 이동 난이도 및 휠체어 접근성 평가
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class DifficultyAnalyzer:
    """클러스터 기반 난이도 분석기 (휠체어 기준)"""

    def __init__(self):
        # 난이도 평가 기준 (휠체어 접근성 기반) - 새로운 특성 이름에 맞게 업데이트
        self.difficulty_criteria = {
            'smoothness': {
                'weight': 0.35,
                'description': '경로 평활성 (휠체어 주행 편의성)',
                'good_threshold': 'low',  # 낮을수록 쉬움
                'features': ['acc_std_mean', 'acc_mean_diff_mean', 'acc_x_std_mean', 'acc_y_std_mean', 'acc_z_std_mean']
            },
            'stability': {
                'weight': 0.25,
                'description': '회전 안정성 (휠체어 균형 유지)',
                'good_threshold': 'low',
                'features': ['gyro_rms_mean', 'gyro_std_mean', 'gyro_x_std_mean', 'gyro_y_std_mean', 'gyro_z_std_mean']
            },
            'shock_resistance': {
                'weight': 0.25,
                'description': '충격 저항성 (장애물 및 단차 대응)',
                'good_threshold': 'low',
                'features': ['jerk_mean_mean', 'jerk_max_mean', 'acc_max_mean']
            },
            'comfort': {
                'weight': 0.15,
                'description': '승차감 (전체적인 편안함)',
                'good_threshold': 'low',
                'features': ['acc_rms_mean', 'acc_range_mean', 'activity_intensity_mean']
            }
        }

    def load_cluster_data(self, clustered_features_path: str = "results/clustered_features.csv") -> pd.DataFrame:
        """클러스터 특성 데이터 로드 및 집계"""
        print(f"📊 클러스터 특성 데이터 로드: {clustered_features_path}")

        # clustered_features.csv 로드
        full_data = pd.read_csv(clustered_features_path)

        # 클러스터별 평균 계산
        cluster_groups = full_data.groupby('cluster')
        cluster_means = cluster_groups.mean()

        # count와 percentage 추가
        cluster_counts = cluster_groups.size()
        total_count = len(full_data)

        self.cluster_data = cluster_means.copy()
        self.cluster_data['cluster'] = cluster_means.index
        self.cluster_data['count'] = cluster_counts.values
        self.cluster_data['percentage'] = (cluster_counts.values / total_count) * 100

        # 특성명에 _mean 추가 (기존 코드 호환성)
        for col in cluster_means.columns:
            if col not in ['cluster', 'count', 'percentage', 'window_id', 'start_idx', 'end_idx', 'window_size']:
                self.cluster_data[f'{col}_mean'] = self.cluster_data[col]

        print(f"✅ 로드 완료: {len(self.cluster_data)}개 클러스터")
        return self.cluster_data

    def normalize_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """특성 정규화 (0-1 스케일)"""
        normalized_df = df.copy()

        for feature in features:
            if feature in df.columns:
                min_val = df[feature].min()
                max_val = df[feature].max()
                if max_val > min_val:
                    normalized_df[f'{feature}_norm'] = (df[feature] - min_val) / (max_val - min_val)
                else:
                    normalized_df[f'{feature}_norm'] = 0.0
            else:
                print(f"⚠️  특성 없음: {feature}")

        return normalized_df

    def calculate_difficulty_score_simple(self, cluster_row: pd.Series) -> Dict:
        """간단한 임계값 기반 난이도 계산"""
        # 핵심 특성값 추출
        acc_std = cluster_row.get('acc_std_mean', 0)
        gyro_std = cluster_row.get('gyro_std_mean', 0) 
        jerk_max = cluster_row.get('jerk_max_mean', 0)
        acc_range = cluster_row.get('acc_range_mean', 0)
        
        # 임계값 기반 점수 계산
        # 1. Smoothness (가속도 안정성)
        if acc_std <= 2.0:
            smoothness = 0.0
        elif acc_std <= 5.0:
            smoothness = 0.3
        elif acc_std <= 8.0:
            smoothness = 0.7
        else:
            smoothness = 1.0
        
        # 2. Stability (회전 안정성)
        if gyro_std <= 0.3:
            stability = 0.0
        elif gyro_std <= 0.7:
            stability = 0.3
        elif gyro_std <= 1.2:
            stability = 0.7
        else:
            stability = 1.0
        
        # 3. Shock resistance (충격 저항성)
        if jerk_max <= 20.0:
            shock = 0.0
        elif jerk_max <= 35.0:
            shock = 0.3
        elif jerk_max <= 50.0:
            shock = 0.7
        else:
            shock = 1.0
        
        # 4. Comfort (편안함)
        if acc_range <= 15.0:
            comfort = 0.0
        elif acc_range <= 30.0:
            comfort = 0.3
        elif acc_range <= 45.0:
            comfort = 0.7
        else:
            comfort = 1.0
        
        # 가중평균
        difficulty = (smoothness * 0.35 + stability * 0.25 + shock * 0.25 + comfort * 0.15)
        
        return {
            'difficulty_score': difficulty,
            'criterion_scores': {
                'smoothness': {'raw_score': smoothness, 'weighted_score': smoothness * 0.35},
                'stability': {'raw_score': stability, 'weighted_score': stability * 0.25},
                'shock_resistance': {'raw_score': shock, 'weighted_score': shock * 0.25},
                'comfort': {'raw_score': comfort, 'weighted_score': comfort * 0.15}
            }
        }

    def calculate_difficulty_score(self, cluster_row: pd.Series) -> Dict:
        """클러스터별 난이도 점수 계산 (간소화된 버전)"""
        # 간단한 버전 사용
        return self.calculate_difficulty_score_simple(cluster_row)


    def classify_difficulty_level(self, difficulty_score: float) -> Dict:
        """난이도 점수를 레벨로 분류"""
        if difficulty_score < 0.2:
            return {'level': 0, 'name': '매우 쉬움', 'color': 'green', 'description': '평지, 매우 안전'}
        elif difficulty_score < 0.4:
            return {'level': 1, 'name': '쉬움', 'color': 'lightgreen', 'description': '완만한 경사, 안전'}
        elif difficulty_score < 0.6:
            return {'level': 2, 'name': '보통', 'color': 'yellow', 'description': '중간 경사, 주의 필요'}
        elif difficulty_score < 0.8:
            return {'level': 3, 'name': '어려움', 'color': 'orange', 'description': '가파른 경사, 위험'}
        else:
            return {'level': 4, 'name': '매우 어려움', 'color': 'red', 'description': '계단/극한, 매우 위험'}


    def analyze_all_clusters(self) -> pd.DataFrame:
        """모든 클러스터 분석"""
        if not hasattr(self, 'cluster_data'):
            raise ValueError("클러스터 데이터를 먼저 로드하세요")

        print("🔍 클러스터별 난이도 분석 시작")

        # 특성 정규화
        all_features = []
        for config in self.difficulty_criteria.values():
            all_features.extend(config['features'])

        unique_features = list(set(all_features))
        normalized_data = self.normalize_features(self.cluster_data, unique_features)

        results = []

        for idx, row in normalized_data.iterrows():
            cluster_id = row['cluster']
            cluster_count = row['count']
            cluster_percentage = row['percentage']

            print(f"\n📊 클러스터 {cluster_id} 분석 ({cluster_count}개 윈도우, {cluster_percentage:.1f}%)")

            # 난이도 분석
            difficulty_result = self.calculate_difficulty_score(row)
            difficulty_level = self.classify_difficulty_level(difficulty_result['difficulty_score'])

            print(f"  난이도: {difficulty_result['difficulty_score']:.3f} ({difficulty_level['name']})")

            result = {
                'cluster': cluster_id,
                'count': cluster_count,
                'percentage': cluster_percentage,
                'difficulty_score': difficulty_result['difficulty_score'],
                'difficulty_level': difficulty_level['level'],
                'difficulty_name': difficulty_level['name'],
                'difficulty_description': difficulty_level['description']
            }

            # 세부 점수 추가
            for criterion, score_info in difficulty_result['criterion_scores'].items():
                result[f'difficulty_{criterion}'] = score_info['weighted_score']

            results.append(result)

        results_df = pd.DataFrame(results)
        return results_df

    def visualize_analysis(self, results_df: pd.DataFrame, save_path: str = None):
        """분석 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 클러스터별 난이도 비교
        ax1 = axes[0, 0]
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        bars1 = ax1.bar(results_df['cluster'], results_df['difficulty_score'])
        for i, (bar, level) in enumerate(zip(bars1, results_df['difficulty_level'])):
            bar.set_color(colors[level])
        ax1.set_title('클러스터별 난이도 점수')
        ax1.set_xlabel('클러스터')
        ax1.set_ylabel('난이도 점수 (0-1)')

        # 2. 클러스터별 세부 기준 점수
        ax2 = axes[0, 1]
        criteria = ['smoothness', 'stability', 'shock_resistance', 'comfort']
        bottom = np.zeros(len(results_df))
        
        for criterion in criteria:
            if f'difficulty_{criterion}' in results_df.columns:
                bars = ax2.bar(results_df['cluster'], results_df[f'difficulty_{criterion}'], 
                              bottom=bottom, label=criterion)
                bottom += results_df[f'difficulty_{criterion}']
        
        ax2.set_title('클러스터별 세부 기준 점수')
        ax2.set_xlabel('클러스터')
        ax2.set_ylabel('점수')
        ax2.legend()

        # 3. 클러스터 분포
        ax3 = axes[1, 0]
        ax3.pie(results_df['percentage'], labels=[f'클러스터 {c}' for c in results_df['cluster']],
               autopct='%1.1f%%', startangle=90)
        ax3.set_title('클러스터 분포')

        # 4. 난이도 레벨 분포
        ax4 = axes[1, 1]
        level_counts = results_df['difficulty_level'].value_counts().sort_index()
        level_names = ['매우 쉬움', '쉬움', '보통', '어려움', '매우 어려움']
        level_colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        
        bars = ax4.bar(range(len(level_counts)), level_counts.values)
        for i, bar in enumerate(bars):
            if i < len(level_colors):
                bar.set_color(level_colors[i])
        
        ax4.set_xlabel('난이도 레벨')
        ax4.set_ylabel('클러스터 수')
        ax4.set_title('난이도 레벨별 클러스터 분포')
        ax4.set_xticks(range(len(level_counts)))
        ax4.set_xticklabels([level_names[i] for i in level_counts.index])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 시각화 저장: {save_path}")

        plt.show()

    def generate_report(self, results_df: pd.DataFrame) -> str:
        """분석 결과 보고서 생성"""
        report = []
        report.append("🎯 클러스터별 난이도 분석 보고서")
        report.append("=" * 60)

        for idx, row in results_df.iterrows():
            report.append(f"\n📊 클러스터 {row.cluster} ({row.count}개 윈도우, {row.percentage:.1f}%)")
            report.append("-" * 40)
            report.append(f"🔥 난이도: {row.difficulty_score:.3f} - {row.difficulty_name}")
            report.append(f"   {row.difficulty_description}")

            # 권장사항 (난이도 기준)
            if row.difficulty_score <= 0.4:
                report.append("✅ 권장: 이동하기 쉬운 경로")
            elif row.difficulty_score <= 0.6:
                report.append("⚠️  주의: 이동 시 조심 필요")
            else:
                report.append("❌ 비권장: 이동이 어려운 경로")

        # 종합 요약
        report.append(f"\n📋 종합 요약")
        report.append("-" * 40)

        easiest_cluster = results_df.loc[results_df['difficulty_score'].idxmin()]
        hardest_cluster = results_df.loc[results_df['difficulty_score'].idxmax()]

        report.append(f"🏆 가장 쉬운 경로: 클러스터 {easiest_cluster.cluster} (난이도 {easiest_cluster.difficulty_score:.3f})")
        report.append(f"⚠️  가장 어려운 경로: 클러스터 {hardest_cluster.cluster} (난이도 {hardest_cluster.difficulty_score:.3f})")

        # 전체 경로 평가
        weighted_difficulty = (results_df['difficulty_score'] * results_df['percentage'] / 100).sum()
        report.append(f"📊 전체 경로 평균 난이도: {weighted_difficulty:.3f}")

        if weighted_difficulty <= 0.4:
            report.append("✅ 전체적으로 이동하기 쉬운 경로입니다")
        elif weighted_difficulty <= 0.6:
            report.append("⚠️  전체적으로 이동 시 주의가 필요한 경로입니다")
        else:
            report.append("❌ 전체적으로 이동이 어려운 경로입니다")

        return "\n".join(report)

def main():
    """메인 실행"""
    print("🎯 클러스터 기반 난이도 분석")
    print("=" * 60)

    # 분석기 생성
    analyzer = DifficultyAnalyzer()

    # 클러스터 데이터 로드
    cluster_data = analyzer.load_cluster_data("results/clustered_features.csv")

    # 전체 분석 수행
    results_df = analyzer.analyze_all_clusters()

    # 시각화
    try:
        import matplotlib
        matplotlib.use('Agg')  # GUI 없이 저장
        analyzer.visualize_analysis(results_df, save_path="results/difficulty_analysis.png")
    except Exception as e:
        print(f"⚠️  시각화 저장 실패: {e}")

    # 보고서 생성
    report = analyzer.generate_report(results_df)
    print(f"\n{report}")

    # 결과 저장
    results_df.to_csv("results/difficulty_analysis.csv", index=False)

    with open("results/difficulty_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n💾 결과 저장:")
    print(f"  - results/difficulty_analysis.csv")
    print(f"  - results/difficulty_report.txt")
    print(f"  - results/difficulty_analysis.png")

if __name__ == "__main__":
    main()