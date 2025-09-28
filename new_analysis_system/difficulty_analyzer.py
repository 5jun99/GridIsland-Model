#!/usr/bin/env python3
"""
난이도 분석기 - 클러스터별 이동 난이도 및 휠체어 접근성 평가
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class DifficultyAnalyzer:
    """클러스터 기반 난이도 분석기"""

    def __init__(self):
        # 난이도 평가 기준 정의
        self.difficulty_criteria = {
            'movement_intensity': {
                'weight': 0.3,
                'description': '전체적인 움직임 강도',
                'features': ['acc_mag_rms_mean', 'gyro_mag_rms_mean', 'activity_intensity_mean']
            },
            'instability': {
                'weight': 0.25,
                'description': '움직임 불안정성 (흔들림)',
                'features': ['acc_mag_std_mean', 'acc_mag_var_mean', 'gyro_mag_std_mean']
            },
            'sudden_changes': {
                'weight': 0.2,
                'description': '급격한 변화 (충격, 진동)',
                'features': ['acc_mag_mean_diff_mean', 'jerk_x_rms_mean', 'jerk_y_rms_mean', 'jerk_z_rms_mean']
            },
            'frequency_patterns': {
                'weight': 0.15,
                'description': '고주파 진동 패턴',
                'features': ['acc_mag_high_freq_energy_mean', 'acc_mag_spectral_centroid_mean']
            },
            'range_variability': {
                'weight': 0.1,
                'description': '움직임 범위 변동성',
                'features': ['acc_mag_range_mean', 'acc_mag_iqr_mean']
            }
        }

        # 휠체어 접근성 평가 기준 (개선된 버전)
        self.wheelchair_criteria = {
            'smoothness': {
                'weight': 0.35,
                'description': '경로 평활성 (휠체어 주행 편의성)',
                'good_threshold': 'low',  # 낮을수록 좋음
                'features': ['acc_mag_std_mean', 'acc_mag_mean_diff_mean', 'acc_mag_var_mean']
            },
            'stability': {
                'weight': 0.25,
                'description': '회전 안정성 (휠체어 균형 유지)',
                'good_threshold': 'low',
                'features': ['gyro_mag_rms_mean', 'gyro_mag_std_mean']
            },
            'shock_resistance': {
                'weight': 0.25,
                'description': '충격 저항성 (장애물 및 단차 대응)',
                'good_threshold': 'low',
                'features': ['jerk_x_rms_mean', 'jerk_y_rms_mean', 'jerk_z_rms_mean']
            },
            'comfort': {
                'weight': 0.15,
                'description': '승차감 (전체적인 편안함)',
                'good_threshold': 'low',
                'features': ['acc_mag_rms_mean', 'acc_mag_range_mean']
            }
        }

    def load_cluster_data(self, cluster_characteristics_path: str) -> pd.DataFrame:
        """클러스터 특성 데이터 로드"""
        print(f"📊 클러스터 특성 데이터 로드: {cluster_characteristics_path}")
        self.cluster_data = pd.read_csv(cluster_characteristics_path)
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

    def calculate_difficulty_score(self, cluster_row: pd.Series) -> Dict:
        """클러스터별 난이도 점수 계산"""
        scores = {}
        total_score = 0.0

        for criterion, config in self.difficulty_criteria.items():
            criterion_score = 0.0
            available_features = 0

            for feature in config['features']:
                if feature in cluster_row.index:
                    # 정규화된 값 사용 (높을수록 어려움)
                    normalized_feature = f'{feature}_norm'
                    if normalized_feature in cluster_row.index:
                        criterion_score += cluster_row[normalized_feature]
                        available_features += 1

            if available_features > 0:
                criterion_score = criterion_score / available_features
                weighted_score = criterion_score * config['weight']
                scores[criterion] = {
                    'raw_score': criterion_score,
                    'weighted_score': weighted_score,
                    'weight': config['weight'],
                    'description': config['description']
                }
                total_score += weighted_score
            else:
                scores[criterion] = {
                    'raw_score': 0.0,
                    'weighted_score': 0.0,
                    'weight': config['weight'],
                    'description': config['description']
                }

        return {
            'total_difficulty': total_score,
            'criterion_scores': scores
        }

    def calculate_wheelchair_accessibility(self, cluster_row: pd.Series) -> Dict:
        """휠체어 접근성 점수 계산"""
        scores = {}
        total_score = 0.0

        for criterion, config in self.wheelchair_criteria.items():
            criterion_score = 0.0
            available_features = 0

            for feature in config['features']:
                if feature in cluster_row.index:
                    normalized_feature = f'{feature}_norm'
                    if normalized_feature in cluster_row.index:
                        # 휠체어 접근성은 낮을수록 좋으므로 1에서 빼기
                        if config['good_threshold'] == 'low':
                            accessibility_score = 1.0 - cluster_row[normalized_feature]
                        else:
                            accessibility_score = cluster_row[normalized_feature]

                        criterion_score += accessibility_score
                        available_features += 1

            if available_features > 0:
                criterion_score = criterion_score / available_features
                weighted_score = criterion_score * config['weight']
                scores[criterion] = {
                    'raw_score': criterion_score,
                    'weighted_score': weighted_score,
                    'weight': config['weight'],
                    'description': config['description']
                }
                total_score += weighted_score
            else:
                scores[criterion] = {
                    'raw_score': 0.0,
                    'weighted_score': 0.0,
                    'weight': config['weight'],
                    'description': config['description']
                }

        return {
            'total_accessibility': total_score,
            'criterion_scores': scores
        }

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

    def classify_wheelchair_accessibility(self, accessibility_score: float) -> Dict:
        """휠체어 접근성 점수를 등급으로 분류"""
        if accessibility_score >= 0.8:
            return {'grade': 'A', 'name': '우수', 'color': 'green', 'description': '휠체어 이용 매우 적합'}
        elif accessibility_score >= 0.6:
            return {'grade': 'B', 'name': '양호', 'color': 'lightgreen', 'description': '휠체어 이용 적합'}
        elif accessibility_score >= 0.4:
            return {'grade': 'C', 'name': '보통', 'color': 'yellow', 'description': '휠체어 이용 가능 (주의)'}
        elif accessibility_score >= 0.2:
            return {'grade': 'D', 'name': '어려움', 'color': 'orange', 'description': '휠체어 이용 어려움'}
        else:
            return {'grade': 'F', 'name': '부적합', 'color': 'red', 'description': '휠체어 이용 불가'}

    def analyze_all_clusters(self) -> pd.DataFrame:
        """모든 클러스터 분석"""
        if not hasattr(self, 'cluster_data'):
            raise ValueError("클러스터 데이터를 먼저 로드하세요")

        print("🔍 클러스터별 난이도 및 접근성 분석 시작")

        # 특성 정규화
        all_features = []
        for config in self.difficulty_criteria.values():
            all_features.extend(config['features'])
        for config in self.wheelchair_criteria.values():
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
            difficulty_level = self.classify_difficulty_level(difficulty_result['total_difficulty'])

            # 휠체어 접근성 분석
            accessibility_result = self.calculate_wheelchair_accessibility(row)
            accessibility_grade = self.classify_wheelchair_accessibility(accessibility_result['total_accessibility'])

            print(f"  난이도: {difficulty_result['total_difficulty']:.3f} ({difficulty_level['name']})")
            print(f"  휠체어 접근성: {accessibility_result['total_accessibility']:.3f} ({accessibility_grade['name']})")

            result = {
                'cluster': cluster_id,
                'count': cluster_count,
                'percentage': cluster_percentage,
                'difficulty_score': difficulty_result['total_difficulty'],
                'difficulty_level': difficulty_level['level'],
                'difficulty_name': difficulty_level['name'],
                'difficulty_description': difficulty_level['description'],
                'wheelchair_score': accessibility_result['total_accessibility'],
                'wheelchair_grade': accessibility_grade['grade'],
                'wheelchair_name': accessibility_grade['name'],
                'wheelchair_description': accessibility_grade['description']
            }

            # 세부 점수 추가
            for criterion, score_info in difficulty_result['criterion_scores'].items():
                result[f'difficulty_{criterion}'] = score_info['weighted_score']

            for criterion, score_info in accessibility_result['criterion_scores'].items():
                result[f'wheelchair_{criterion}'] = score_info['weighted_score']

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

        # 2. 클러스터별 휠체어 접근성 비교
        ax2 = axes[0, 1]
        grade_colors = {'A': 'green', 'B': 'lightgreen', 'C': 'yellow', 'D': 'orange', 'F': 'red'}
        bars2 = ax2.bar(results_df['cluster'], results_df['wheelchair_score'])
        for i, (bar, grade) in enumerate(zip(bars2, results_df['wheelchair_grade'])):
            bar.set_color(grade_colors.get(grade, 'gray'))
        ax2.set_title('클러스터별 휠체어 접근성')
        ax2.set_xlabel('클러스터')
        ax2.set_ylabel('접근성 점수 (0-1)')

        # 3. 클러스터 분포
        ax3 = axes[1, 0]
        ax3.pie(results_df['percentage'], labels=[f'클러스터 {c}' for c in results_df['cluster']],
               autopct='%1.1f%%', startangle=90)
        ax3.set_title('클러스터 분포')

        # 4. 난이도 vs 접근성 산점도
        ax4 = axes[1, 1]
        scatter = ax4.scatter(results_df['difficulty_score'], results_df['wheelchair_score'],
                            s=results_df['percentage']*10, alpha=0.7, c=results_df['cluster'],
                            cmap='tab10')
        ax4.set_xlabel('난이도 점수')
        ax4.set_ylabel('휠체어 접근성 점수')
        ax4.set_title('난이도 vs 접근성 (크기=분포비율)')

        # 클러스터 번호 표시
        for idx, row in results_df.iterrows():
            ax4.annotate(f'C{row.cluster}',
                        (row.difficulty_score, row.wheelchair_score),
                        xytext=(5, 5), textcoords='offset points')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 시각화 저장: {save_path}")

        plt.show()

    def generate_report(self, results_df: pd.DataFrame) -> str:
        """분석 결과 보고서 생성"""
        report = []
        report.append("🎯 클러스터별 난이도 및 휠체어 접근성 분석 보고서")
        report.append("=" * 60)

        for idx, row in results_df.iterrows():
            report.append(f"\n📊 클러스터 {row.cluster} ({row.count}개 윈도우, {row.percentage:.1f}%)")
            report.append("-" * 40)
            report.append(f"🔥 난이도: {row.difficulty_score:.3f} - {row.difficulty_name}")
            report.append(f"   {row.difficulty_description}")
            report.append(f"♿ 휠체어 접근성: {row.wheelchair_score:.3f} - {row.wheelchair_grade}등급 ({row.wheelchair_name})")
            report.append(f"   {row.wheelchair_description}")

            # 권장사항
            if row.wheelchair_score >= 0.6:
                report.append("✅ 권장: 휠체어 이용 적합한 경로")
            elif row.wheelchair_score >= 0.4:
                report.append("⚠️  주의: 휠체어 이용 시 조심 필요")
            else:
                report.append("❌ 비권장: 휠체어 이용 피하는 것이 좋음")

        # 종합 요약
        report.append(f"\n📋 종합 요약")
        report.append("-" * 40)

        best_cluster = results_df.loc[results_df['wheelchair_score'].idxmax()]
        worst_cluster = results_df.loc[results_df['wheelchair_score'].idxmin()]

        report.append(f"🏆 가장 휠체어 친화적: 클러스터 {best_cluster.cluster} (접근성 {best_cluster.wheelchair_score:.3f})")
        report.append(f"⚠️  가장 주의 필요: 클러스터 {worst_cluster.cluster} (접근성 {worst_cluster.wheelchair_score:.3f})")

        # 전체 경로 평가
        weighted_accessibility = (results_df['wheelchair_score'] * results_df['percentage'] / 100).sum()
        report.append(f"📊 전체 경로 휠체어 접근성: {weighted_accessibility:.3f}")

        if weighted_accessibility >= 0.6:
            report.append("✅ 전체적으로 휠체어 이용에 적합한 경로입니다")
        elif weighted_accessibility >= 0.4:
            report.append("⚠️  전체적으로 휠체어 이용 시 주의가 필요한 경로입니다")
        else:
            report.append("❌ 전체적으로 휠체어 이용이 어려운 경로입니다")

        return "\n".join(report)

def main():
    """메인 실행"""
    print("🎯 클러스터 난이도 및 휠체어 접근성 분석")
    print("=" * 60)

    # 분석기 생성
    analyzer = DifficultyAnalyzer()

    # 클러스터 데이터 로드
    cluster_data = analyzer.load_cluster_data("results/cluster_characteristics.csv")

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