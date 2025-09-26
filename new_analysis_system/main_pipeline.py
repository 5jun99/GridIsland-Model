#!/usr/bin/env python3
"""
완전한 분석 파이프라인
데이터 로딩 → 특성 추출 → 클러스터링 → 난이도 평가 → 휠체어 접근성 분석
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
    """완전한 분석 파이프라인"""
    print("🌊 Grid Island - 완전한 Test 데이터 분석 시스템")
    print("🎯 특성 추출 → 클러스터링 → 난이도 평가 → 휠체어 접근성 분석")
    print("=" * 70)

    # 결과 폴더 생성
    os.makedirs("results", exist_ok=True)

    # 1단계: 데이터 로딩
    print("\n📊 1단계: 센서 데이터 로딩")
    print("-" * 40)

    data_dir = "data/test 2025-09-22 18-30-21"
    sensor_data = load_sensor_data(data_dir)

    if not sensor_data:
        print("❌ 센서 데이터 로딩 실패")
        return

    combined_df = combine_sensor_data(sensor_data)
    data_info = get_data_info(combined_df)

    print(f"📋 데이터 정보:")
    for key, value in data_info.items():
        if key != 'columns':
            print(f"  {key}: {value}")

    # 2단계: 특성 추출
    print("\n🔍 2단계: 특성 추출")
    print("-" * 40)

    extractor = FeatureExtractor(window_size=200, overlap_ratio=0.75)
    features_df, window_positions = extractor.process_data(combined_df)

    # 특성 데이터 저장
    features_path = "results/extracted_features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"💾 특성 데이터 저장: {features_path}")

    print(f"📊 추출 결과:")
    print(f"  윈도우 수: {len(features_df)}")
    print(f"  특성 수: {len(features_df.columns)}")

    # 3단계: 클러스터링 분석
    print("\n🎯 3단계: 클러스터링 분석")
    print("-" * 40)

    analyzer = ClusteringAnalyzer()
    analyzer.load_features(features_path)

    # 전처리
    features_scaled = analyzer.preprocess_features()

    # PCA 적용
    features_pca = analyzer.apply_pca(n_components=10)

    # 최적 클러스터 수 탐색
    print("\n🔍 최적 클러스터 수 탐색:")
    optimization_results = analyzer.find_optimal_clusters(max_k=8, method='kmeans')

    # 추천 클러스터 수 결정
    silhouette_scores = optimization_results['silhouette']
    optimal_k = optimization_results['k_values'][np.argmax(silhouette_scores)]
    print(f"📈 추천 클러스터 수: {optimal_k} (Silhouette Score: {max(silhouette_scores):.3f})")

    # 클러스터링 수행
    print(f"\n🎯 클러스터링 수행 (K={optimal_k}):")
    labels = analyzer.perform_clustering(n_clusters=optimal_k, method='kmeans')

    # 클러스터 특성 분석
    cluster_characteristics = analyzer.analyze_cluster_characteristics(labels)

    # 클러스터링 시각화
    try:
        import matplotlib
        matplotlib.use('Agg')
        analyzer.visualize_clusters(labels, save_path="results/clustering_visualization.png")
    except Exception as e:
        print(f"⚠️  클러스터링 시각화 저장 실패: {e}")

    # 결과 저장
    result_df = features_df.copy()
    result_df['cluster'] = labels
    result_df.to_csv("results/clustered_features.csv", index=False)
    cluster_characteristics.to_csv("results/cluster_characteristics.csv", index=False)

    # 4단계: 난이도 및 휠체어 접근성 분석
    print("\n🎯 4단계: 난이도 및 휠체어 접근성 분석")
    print("-" * 50)

    difficulty_analyzer = DifficultyAnalyzer()
    difficulty_analyzer.load_cluster_data("results/cluster_characteristics.csv")

    # 난이도 분석 수행
    difficulty_results = difficulty_analyzer.analyze_all_clusters()

    # 난이도 분석 시각화
    try:
        difficulty_analyzer.visualize_analysis(difficulty_results, save_path="results/difficulty_analysis.png")
    except Exception as e:
        print(f"⚠️  난이도 분석 시각화 저장 실패: {e}")

    # 보고서 생성
    report = difficulty_analyzer.generate_report(difficulty_results)

    # 결과 저장
    difficulty_results.to_csv("results/difficulty_analysis.csv", index=False)
    with open("results/difficulty_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # 5단계: 종합 요약
    print("\n📋 5단계: 종합 분석 요약")
    print("-" * 40)

    print(f"✅ 분석 완료!")
    print(f"\n📊 데이터 규모:")
    print(f"  원본 센서 데이터: {len(combined_df):,}개 샘플")
    print(f"  측정 시간: {data_info['duration_seconds']:.1f}초")
    print(f"  샘플링 주파수: {data_info['sampling_rate_hz']:.1f}Hz")

    print(f"\n🔍 특성 추출 결과:")
    print(f"  윈도우 수: {len(features_df):,}개")
    print(f"  추출된 특성: {len(features_df.columns)}개")

    print(f"\n🎯 클러스터링 결과:")
    print(f"  최적 클러스터 수: {optimal_k}개")
    print(f"  Silhouette Score: {max(silhouette_scores):.3f}")

    print(f"\n🏥 휠체어 접근성 평가:")
    for idx, row in difficulty_results.iterrows():
        status_emoji = "✅" if row.wheelchair_score >= 0.6 else "⚠️" if row.wheelchair_score >= 0.4 else "❌"
        print(f"  {status_emoji} 클러스터 {row.cluster}: {row.wheelchair_grade}등급 ({row.wheelchair_score:.3f}) - {row.wheelchair_name}")

    # 전체 경로 평가
    weighted_accessibility = (difficulty_results['wheelchair_score'] * difficulty_results['percentage'] / 100).sum()
    overall_emoji = "✅" if weighted_accessibility >= 0.6 else "⚠️" if weighted_accessibility >= 0.4 else "❌"
    print(f"\n🎯 전체 경로 평가:")
    print(f"  {overall_emoji} 종합 휠체어 접근성: {weighted_accessibility:.3f}")

    if weighted_accessibility >= 0.6:
        recommendation = "휠체어 이용에 적합한 경로입니다"
    elif weighted_accessibility >= 0.4:
        recommendation = "휠체어 이용 시 주의가 필요한 경로입니다"
    else:
        recommendation = "휠체어 이용이 어려운 경로입니다"

    print(f"  💡 권장사항: {recommendation}")

    # 클러스터별 상세 정보 표시
    print(f"\n📊 클러스터별 상세 정보:")
    for idx, row in difficulty_results.iterrows():
        print(f"\n  🔸 클러스터 {row.cluster} ({row.count}개 윈도우, {row.percentage:.1f}%):")
        print(f"     난이도: {row.difficulty_name} ({row.difficulty_score:.3f})")
        print(f"     휠체어 접근성: {row.wheelchair_grade}등급 - {row.wheelchair_name}")
        print(f"     설명: {row.difficulty_description}")

    print(f"\n📁 저장된 결과 파일:")
    result_files = [
        "results/extracted_features.csv",
        "results/clustered_features.csv",
        "results/cluster_characteristics.csv",
        "results/difficulty_analysis.csv",
        "results/difficulty_report.txt",
        "results/clustering_visualization.png",
        "results/difficulty_analysis.png"
    ]

    for file in result_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")

    print(f"\n🎉 Grid Island 완전 분석 완료!")
    print(f"📖 상세한 분석 결과는 results/difficulty_report.txt를 확인하세요")

    # 분석 결과 출력
    print(f"\n{report}")

if __name__ == "__main__":
    main()