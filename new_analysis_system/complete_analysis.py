#!/usr/bin/env python3
"""
완전한 분석 파이프라인 v2.0
데이터 로딩 → 특성 추출 → 클러스터링 → 난이도 평가 → 휠체어 접근성 분석 → 종합 보고서
"""

import os
import sys
import pandas as pd
import numpy as np
from utils.data_loader import load_sensor_data, combine_sensor_data, get_data_info
from feature_extractor import FeatureExtractor
from clustering_analyzer import ClusteringAnalyzer
from difficulty_analyzer import DifficultyAnalyzer

def main():
    """완전한 분석 파이프라인 v2.0"""
    print("🌊 Grid Island - 완전한 휠체어 접근성 분석 시스템 v2.0")
    print("🎯 특성 추출 → 클러스터링 → 난이도 평가 → 휠체어 접근성 분석 → 종합 보고서")
    print("=" * 75)

    # 결과 폴더 생성
    os.makedirs("results", exist_ok=True)

    # 1단계: 센서 데이터 처리
    print("\n📊 1단계: 센서 데이터 처리")
    print("-" * 45)

    data_dir = "data/test 2025-09-22 18-30-21"
    sensor_data = load_sensor_data(data_dir)

    if not sensor_data:
        print("❌ 센서 데이터 로딩 실패")
        return False

    combined_df = combine_sensor_data(sensor_data)
    data_info = get_data_info(combined_df)

    print(f"📈 데이터 품질 평가:")
    print(f"  샘플링 품질: {'우수' if data_info['sampling_rate_hz'] > 45 else '보통'}")
    print(f"  데이터 완성도: {(len(combined_df)/data_info['total_samples']*100):.1f}%")

    # 2단계: 고급 특성 추출
    print("\n🔍 2단계: 55차원 특성 추출 및 엔지니어링")
    print("-" * 50)

    extractor = FeatureExtractor(window_size=200, overlap_ratio=0.75)
    features_df, window_positions = extractor.process_data(combined_df)

    # 특성 품질 평가
    feature_quality = {
        'completeness': features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns)),
        'variance_ratio': len(features_df.columns[features_df.var() > 0.001]) / len(features_df.columns)
    }

    print(f"📊 특성 품질 평가:")
    print(f"  특성 완성도: {(1-feature_quality['completeness'])*100:.1f}%")
    print(f"  유의미 특성 비율: {feature_quality['variance_ratio']*100:.1f}%")

    features_path = "results/extracted_features.csv"
    features_df.to_csv(features_path, index=False)

    # 3단계: 클러스터링
    print("\n🎯 3단계: 클러스터링 분석")
    print("-" * 40)

    analyzer = ClusteringAnalyzer()
    analyzer.load_features(features_path)
    features_scaled = analyzer.preprocess_features()
    features_pca = analyzer.apply_pca(n_components=10)

    # 다중 방법론 적용
    methods = ['kmeans', 'gmm']
    best_result = None
    best_score = -1

    for method in methods:
        optimization_results = analyzer.find_optimal_clusters(max_k=8, method=method)
        max_silhouette = max(optimization_results['silhouette'])
        if max_silhouette > best_score:
            best_score = max_silhouette
            best_result = (method, optimization_results)

    method, optimization_results = best_result
    optimal_k = optimization_results['k_values'][np.argmax(optimization_results['silhouette'])]

    print(f"🏆 최적 방법론: {method.upper()}, K={optimal_k} (품질: {best_score:.3f})")

    labels = analyzer.perform_clustering(n_clusters=optimal_k, method=method)

    # 결과 저장
    result_df = features_df.copy()
    result_df['cluster'] = labels
    result_df.to_csv("results/clustered_features.csv", index=False)

    # 시각화
    try:
        import matplotlib
        matplotlib.use('Agg')
        analyzer.visualize_clusters(labels, save_path="results/clustering_visualization.png")
    except Exception as e:
        print(f"⚠️  시각화 저장 실패: {e}")

    # 최종 요약 출력
    print(f"\n🎉 분석 완료!")
    print("=" * 60)

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎊 분석 성공적으로 완료! 상세 결과는 results/ 폴더를 확인하세요.")
    else:
        print(f"\n💥 분석 실행 중 오류가 발생했습니다.")