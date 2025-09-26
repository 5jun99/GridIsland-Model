#!/usr/bin/env python3
"""
클러스터링 분석기 - 추출된 특성으로 클러스터링 수행
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class ClusteringAnalyzer:
    """클러스터링 분석기"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.features_df = None
        self.features_scaled = None
        self.features_pca = None

    def load_features(self, features_path: str) -> pd.DataFrame:
        """특성 데이터 로드"""
        print(f"📊 특성 데이터 로드: {features_path}")
        self.features_df = pd.read_csv(features_path)
        print(f"✅ 로드 완료: {len(self.features_df)} 윈도우, {len(self.features_df.columns)} 특성")
        return self.features_df

    def preprocess_features(self, exclude_cols: list = None) -> np.ndarray:
        """특성 전처리 (스케일링, 차원축소)"""
        if self.features_df is None:
            raise ValueError("특성 데이터를 먼저 로드하세요")

        # 제외할 컬럼들 (메타데이터)
        if exclude_cols is None:
            exclude_cols = ['window_id', 'start_idx', 'end_idx', 'window_size']

        # 숫자형 특성만 선택
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        print(f"🔧 전처리할 특성: {len(feature_cols)}개")

        # 결측값 처리
        features_clean = self.features_df[feature_cols].fillna(0)

        # 무한대값 처리
        features_clean = features_clean.replace([np.inf, -np.inf], 0)

        # 표준화
        self.features_scaled = self.scaler.fit_transform(features_clean)
        print(f"✅ 표준화 완료: {self.features_scaled.shape}")

        return self.features_scaled

    def apply_pca(self, n_components: int = 10) -> np.ndarray:
        """PCA 차원 축소"""
        if self.features_scaled is None:
            raise ValueError("특성을 먼저 전처리하세요")

        self.pca = PCA(n_components=n_components)
        self.features_pca = self.pca.fit_transform(self.features_scaled)

        # 설명 분산 출력
        explained_var = self.pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        print(f"📊 PCA 결과 ({n_components}개 컴포넌트):")
        print(f"  총 설명 분산: {cumulative_var[-1]:.3f}")
        print(f"  각 컴포넌트 기여도: {explained_var[:5]}")

        return self.features_pca

    def find_optimal_clusters(self, max_k: int = 10, method: str = 'kmeans') -> dict:
        """최적 클러스터 수 찾기"""
        if self.features_scaled is None:
            raise ValueError("특성을 먼저 전처리하세요")

        print(f"🔍 최적 클러스터 수 탐색 (K=2~{max_k}, 방법: {method})")

        results = {
            'k_values': [],
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }

        for k in range(2, max_k + 1):
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            elif method == 'gmm':
                clusterer = GaussianMixture(n_components=k, random_state=42)
            else:
                raise ValueError("지원하지 않는 방법입니다")

            labels = clusterer.fit_predict(self.features_scaled)

            # 평가 지표 계산
            silhouette = silhouette_score(self.features_scaled, labels)
            calinski = calinski_harabasz_score(self.features_scaled, labels)
            davies_bouldin = davies_bouldin_score(self.features_scaled, labels)

            results['k_values'].append(k)
            results['silhouette'].append(silhouette)
            results['calinski_harabasz'].append(calinski)
            results['davies_bouldin'].append(davies_bouldin)

            if method == 'kmeans':
                results['inertia'].append(clusterer.inertia_)

            print(f"  K={k}: Silhouette={silhouette:.3f}, Calinski={calinski:.1f}, Davies-Bouldin={davies_bouldin:.3f}")

        return results

    def perform_clustering(self, n_clusters: int = 5, method: str = 'kmeans') -> np.ndarray:
        """클러스터링 수행"""
        if self.features_scaled is None:
            raise ValueError("특성을 먼저 전처리하세요")

        print(f"🎯 클러스터링 수행: {method}, K={n_clusters}")

        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'gmm':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError("지원하지 않는 방법입니다")

        labels = clusterer.fit_predict(self.features_scaled)

        # 클러스터링 결과 평가
        if len(set(labels)) > 1:  # 클러스터가 2개 이상인 경우만
            silhouette = silhouette_score(self.features_scaled, labels)
            calinski = calinski_harabasz_score(self.features_scaled, labels)
            davies_bouldin = davies_bouldin_score(self.features_scaled, labels)

            print(f"📊 클러스터링 결과:")
            print(f"  클러스터 수: {len(set(labels))}")
            print(f"  Silhouette Score: {silhouette:.3f}")
            print(f"  Calinski-Harabasz Index: {calinski:.1f}")
            print(f"  Davies-Bouldin Index: {davies_bouldin:.3f}")

            # 클러스터별 분포
            unique, counts = np.unique(labels, return_counts=True)
            print(f"  클러스터 분포:")
            for cluster, count in zip(unique, counts):
                print(f"    클러스터 {cluster}: {count}개 ({count/len(labels)*100:.1f}%)")

        return labels

    def visualize_clusters(self, labels: np.ndarray, save_path: str = None):
        """클러스터링 결과 시각화"""
        if self.features_scaled is None:
            raise ValueError("특성을 먼저 전처리하세요")

        # t-SNE로 2D 시각화
        print("📊 t-SNE 시각화 생성 중...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1))
        features_2d = tsne.fit_transform(self.features_scaled)

        # 플롯 생성
        plt.figure(figsize=(12, 8))

        # 서브플롯 1: t-SNE 결과
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.title('t-SNE Visualization of Clusters')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(scatter)

        # 서브플롯 2: PCA가 있으면 PCA 결과
        if self.features_pca is not None:
            plt.subplot(2, 2, 2)
            scatter = plt.scatter(self.features_pca[:, 0], self.features_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
            plt.title('PCA Visualization of Clusters')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.colorbar(scatter)

        # 서브플롯 3: 클러스터 분포
        plt.subplot(2, 2, 3)
        unique, counts = np.unique(labels, return_counts=True)
        plt.bar(unique, counts, alpha=0.7)
        plt.title('Cluster Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Count')

        # 서브플롯 4: 시간에 따른 클러스터 변화
        plt.subplot(2, 2, 4)
        plt.plot(labels, alpha=0.7)
        plt.title('Clusters over Time')
        plt.xlabel('Window Index')
        plt.ylabel('Cluster')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 시각화 저장: {save_path}")

        plt.show()

    def analyze_cluster_characteristics(self, labels: np.ndarray) -> pd.DataFrame:
        """클러스터별 특성 분석"""
        if self.features_df is None:
            raise ValueError("특성 데이터를 먼저 로드하세요")

        # 클러스터 라벨 추가
        analysis_df = self.features_df.copy()
        analysis_df['cluster'] = labels

        # 숫자형 컬럼만 선택
        numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['window_id', 'start_idx', 'end_idx', 'window_size', 'cluster']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # 클러스터별 통계
        cluster_stats = []

        for cluster in sorted(analysis_df['cluster'].unique()):
            cluster_data = analysis_df[analysis_df['cluster'] == cluster]
            stats = {
                'cluster': cluster,
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(analysis_df) * 100
            }

            # 주요 특성들의 평균값
            for col in feature_cols[:20]:  # 상위 20개 특성
                stats[f'{col}_mean'] = cluster_data[col].mean()

            cluster_stats.append(stats)

        cluster_stats_df = pd.DataFrame(cluster_stats)
        return cluster_stats_df

def main():
    """메인 실행 함수"""
    print("🎯 클러스터링 분석 시작")
    print("=" * 60)

    # 분석기 생성
    analyzer = ClusteringAnalyzer()

    # 특성 데이터 로드
    features_df = analyzer.load_features("results/extracted_features.csv")

    # 전처리
    features_scaled = analyzer.preprocess_features()

    # PCA 적용
    features_pca = analyzer.apply_pca(n_components=10)

    # 최적 클러스터 수 찾기
    optimization_results = analyzer.find_optimal_clusters(max_k=8, method='kmeans')

    # 클러스터링 수행 (K=5로 설정)
    labels = analyzer.perform_clustering(n_clusters=5, method='kmeans')

    # 결과 시각화
    analyzer.visualize_clusters(labels, save_path="results/clustering_visualization.png")

    # 클러스터 특성 분석
    cluster_characteristics = analyzer.analyze_cluster_characteristics(labels)

    # 결과 저장
    import os
    os.makedirs("results", exist_ok=True)

    # 라벨이 포함된 데이터 저장
    result_df = features_df.copy()
    result_df['cluster'] = labels
    result_df.to_csv("results/clustered_features.csv", index=False)

    # 클러스터 특성 저장
    cluster_characteristics.to_csv("results/cluster_characteristics.csv", index=False)

    print(f"\n✅ 분석 완료!")
    print(f"📊 결과 파일:")
    print(f"  - results/clustered_features.csv")
    print(f"  - results/cluster_characteristics.csv")
    print(f"  - results/clustering_visualization.png")

if __name__ == "__main__":
    main()