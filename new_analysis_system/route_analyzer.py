#!/usr/bin/env python3
"""
경로 분석기 - 시뮬레이션된 경로들의 난이도 분석 및 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from gps_loader import EnhancedGPSLoader

class RouteAnalyzer:
    """시뮬레이션된 여러 경로의 난이도 분석 및 비교"""
    
    def __init__(self):
        self.routes_data = {}
        self.analysis_results = {}
        
    def load_simulated_routes(self, routes_dir="data/simulated_routes"):
        """시뮬레이션된 경로들 로드"""
        routes_dir = Path(routes_dir)
        
        # 경로 요약 정보 로드
        summary_file = routes_dir / "route_summary.csv"
        if not summary_file.exists():
            raise FileNotFoundError(f"경로 요약 파일이 없습니다: {summary_file}")
        
        summary_df = pd.read_csv(summary_file)
        
        print(f"📊 {len(summary_df)}개 시뮬레이션 경로 분석 시작")
        print("=" * 60)
        
        for _, route in summary_df.iterrows():
            route_path = Path(route['data_path'])
            route_id = route['route_id']
            
            print(f"🛣️  {route['name']} 분석 중...")
            print(f"   종류: {route['difficulty_type']}")
            print(f"   거리: {route['distance_m']}m")
            print(f"   시간: {route['duration_s']/60:.1f}분")
            
            try:
                # GPS 로더로 분석
                loader = EnhancedGPSLoader(str(route_path))
                
                # 1. 데이터 로드
                gps_data = loader.load_gps_data()
                sensor_data = loader.load_sensor_data()
                
                # 2. 동기화
                synchronized_data = loader.synchronize_data(window_size=1.0)
                
                # 3. 특성 추출
                features = loader.extract_advanced_features(window_size=150, overlap_ratio=0.6)
                
                # 4. 클러스터링
                clustered_data = loader.perform_clustering(n_clusters=4)  # 고정 4개
                
                # 5. 난이도 분석
                difficulty_results = loader.analyze_difficulty()
                
                # 6. GPS 매핑
                gps_with_difficulty = loader.map_difficulty_to_gps()
                
                # 결과 저장
                self.routes_data[route_id] = {
                    'info': route.to_dict(),
                    'gps_data': gps_data,
                    'sensor_data': sensor_data,
                    'synchronized_data': synchronized_data,
                    'features': features,
                    'clustered_data': clustered_data,
                    'difficulty_results': difficulty_results,
                    'gps_with_difficulty': gps_with_difficulty
                }
                
                print(f"   ✅ 분석 완료: {len(features)}개 윈도우, {len(features.columns)}개 특성")
                
            except Exception as e:
                print(f"   ❌ 분석 실패: {e}")
                continue
            
            print()
        
        return self.routes_data
    
    def compare_route_difficulties(self):
        """경로별 난이도 비교 분석"""
        if not self.routes_data:
            raise ValueError("경로 데이터를 먼저 로드하세요")
        
        print("🔍 경로별 난이도 비교 분석")
        print("=" * 60)
        
        comparison_results = []
        
        for route_id, route_data in self.routes_data.items():
            route_info = route_data['info']
            difficulty_results = route_data['difficulty_results']
            gps_with_difficulty = route_data['gps_with_difficulty']
            
            # 전체 경로 난이도 통계
            avg_difficulty = gps_with_difficulty['difficulty'].mean()
            std_difficulty = gps_with_difficulty['difficulty'].std()
            max_difficulty = gps_with_difficulty['difficulty'].max()
            min_difficulty = gps_with_difficulty['difficulty'].min()
            
            # 클러스터별 분포
            cluster_distribution = gps_with_difficulty['cluster_id'].value_counts().sort_index()
            
            result = {
                'route_id': route_id,
                'name': route_info['name'],
                'type': route_info['difficulty_type'],
                'distance_m': route_info['distance_m'],
                'duration_min': route_info['duration_s'] / 60,
                'avg_difficulty': avg_difficulty,
                'std_difficulty': std_difficulty,
                'max_difficulty': max_difficulty,
                'min_difficulty': min_difficulty,
                'difficulty_range': max_difficulty - min_difficulty,
                'cluster_distribution': cluster_distribution.to_dict()
            }
            
            comparison_results.append(result)
            
            print(f"📊 {route_info['name']}")
            print(f"   평균 난이도: {avg_difficulty:.3f} ± {std_difficulty:.3f}")
            print(f"   난이도 범위: {min_difficulty:.3f} ~ {max_difficulty:.3f}")
            print(f"   클러스터 분포: {dict(cluster_distribution)}")
            print()
        
        self.analysis_results['comparison'] = comparison_results
        return comparison_results
    
    def create_comprehensive_comparison(self, save_path="results/route_comparison.png"):
        """종합 비교 시각화"""
        if 'comparison' not in self.analysis_results:
            self.compare_route_difficulties()
        
        comparison_data = self.analysis_results['comparison']
        
        # DataFrame 변환
        df = pd.DataFrame(comparison_data)
        
        # 시각화
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 평균 난이도 비교
        ax1 = axes[0, 0]
        bars = ax1.bar(df['name'], df['avg_difficulty'], 
                       color=['green', 'yellow', 'orange', 'red'][:len(df)])
        ax1.set_title('경로별 평균 난이도', fontsize=14, fontweight='bold')
        ax1.set_ylabel('난이도 점수')
        ax1.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, avg in zip(bars, df['avg_difficulty']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{avg:.3f}', ha='center', va='bottom')
        
        # 2. 난이도 범위 비교
        ax2 = axes[0, 1]
        ax2.bar(df['name'], df['difficulty_range'],
                color=['green', 'yellow', 'orange', 'red'][:len(df)])
        ax2.set_title('경로별 난이도 변동폭', fontsize=14, fontweight='bold')
        ax2.set_ylabel('난이도 범위')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 거리 vs 난이도
        ax3 = axes[0, 2]
        scatter = ax3.scatter(df['distance_m'], df['avg_difficulty'], 
                             s=100, c=df['avg_difficulty'], cmap='RdYlGn_r')
        ax3.set_xlabel('거리 (m)')
        ax3.set_ylabel('평균 난이도')
        ax3.set_title('거리 vs 난이도', fontsize=14, fontweight='bold')
        
        # 경로명 라벨
        for i, row in df.iterrows():
            ax3.annotate(row['type'], (row['distance_m'], row['avg_difficulty']),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # 4. 시간 효율성
        ax4 = axes[1, 0]
        efficiency = df['distance_m'] / df['duration_min']  # m/분
        ax4.bar(df['name'], efficiency,
                color=['green', 'yellow', 'orange', 'red'][:len(df)])
        ax4.set_title('경로별 이동 효율성', fontsize=14, fontweight='bold')
        ax4.set_ylabel('속도 (m/분)')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. 난이도 박스플롯 (각 경로의 분포)
        ax5 = axes[1, 1]
        route_difficulties = []
        route_labels = []
        
        for route_id, route_data in self.routes_data.items():
            difficulties = route_data['gps_with_difficulty']['difficulty']
            route_difficulties.append(difficulties)
            route_labels.append(route_data['info']['difficulty_type'])
        
        ax5.boxplot(route_difficulties, labels=route_labels)
        ax5.set_title('경로별 난이도 분포', fontsize=14, fontweight='bold')
        ax5.set_ylabel('난이도 점수')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. 추천 점수 (안전성 기반)
        ax6 = axes[1, 2]
        # 안전성 점수 = 1 / (1 + 평균_난이도 + 난이도_변동폭)
        safety_scores = 1 / (1 + df['avg_difficulty'] + df['difficulty_range'])
        bars = ax6.bar(df['name'], safety_scores,
                       color=['green', 'yellow', 'orange', 'red'][:len(df)])
        ax6.set_title('휠체어 추천 점수', fontsize=14, fontweight='bold')
        ax6.set_ylabel('추천 점수 (높을수록 좋음)')
        ax6.tick_params(axis='x', rotation=45)
        
        # 최고 점수 표시
        best_idx = safety_scores.idxmax()
        best_bar = bars[best_idx]
        ax6.text(best_bar.get_x() + best_bar.get_width()/2, 
                 best_bar.get_height() + 0.01,
                 '★ 추천', ha='center', va='bottom', 
                 fontweight='bold', color='red')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 종합 비교 차트 저장: {save_path}")
        plt.show()
    
    def generate_route_recommendations(self):
        """경로 추천 보고서 생성"""
        if 'comparison' not in self.analysis_results:
            self.compare_route_difficulties()
        
        comparison_data = self.analysis_results['comparison']
        df = pd.DataFrame(comparison_data)
        
        # 추천 점수 계산
        df['safety_score'] = 1 / (1 + df['avg_difficulty'] + df['difficulty_range'])
        df['efficiency_score'] = df['distance_m'] / df['duration_min'] / 100  # 정규화
        df['total_score'] = df['safety_score'] * 0.7 + df['efficiency_score'] * 0.3
        
        # 정렬
        df_sorted = df.sort_values('total_score', ascending=False)
        
        report = []
        report.append("🚀 시뮬레이션 경로 분석 보고서")
        report.append("=" * 60)
        report.append(f"📅 분석 경로 수: {len(df)}개")
        report.append(f"🎯 분석 목표: 휠체어 사용자 최적 경로 찾기")
        report.append("")
        
        report.append("📊 경로별 난이도 분석 결과:")
        for _, route in df_sorted.iterrows():
            difficulty_level = "매우 쉬움" if route['avg_difficulty'] < 0.2 else \
                             "쉬움" if route['avg_difficulty'] < 0.4 else \
                             "보통" if route['avg_difficulty'] < 0.6 else \
                             "어려움" if route['avg_difficulty'] < 0.8 else "매우 어려움"
            
            report.append(f"\n🛣️  {route['name']}")
            report.append(f"   ├─ 경로 유형: {route['type']}")
            report.append(f"   ├─ 거리: {route['distance_m']}m")
            report.append(f"   ├─ 예상 시간: {route['duration_min']:.1f}분")
            report.append(f"   ├─ 평균 난이도: {route['avg_difficulty']:.3f} ({difficulty_level})")
            report.append(f"   ├─ 난이도 변동: {route['difficulty_range']:.3f}")
            report.append(f"   └─ 추천 점수: {route['total_score']:.3f}/1.0")
        
        # 추천 순위
        report.append("\n🏆 휠체어 사용자 추천 순위:")
        for i, (_, route) in enumerate(df_sorted.iterrows(), 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            report.append(f"   {emoji} {route['name']}")
            
            if i == 1:
                report.append(f"      → 가장 안전하고 이용하기 쉬운 경로!")
        
        # 경로 선택 가이드
        report.append("\n💡 경로 선택 가이드:")
        report.append("   🟢 매우 쉬움 (0.0-0.2): 평지, 부드러운 아스팔트")
        report.append("   🟡 쉬움 (0.2-0.4): 완만한 경사, 안정적")
        report.append("   🟠 보통 (0.4-0.6): 약간 가파른 경사")
        report.append("   🔴 어려움 (0.6-0.8): 가파른 언덕, 주의 필요")
        report.append("   ⚫ 매우 어려움 (0.8-1.0): 계단, 극한 경사")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # 파일로 저장
        with open("results/simulation_analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n💾 분석 보고서 저장: results/simulation_analysis_report.txt")
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🔍 시뮬레이션 경로 분석기")
    print("=" * 60)
    
    try:
        analyzer = RouteAnalyzer()
        
        # 1. 시뮬레이션 경로들 로드 및 분석
        routes_data = analyzer.load_simulated_routes()
        
        # 2. 경로별 난이도 비교
        comparison_results = analyzer.compare_route_difficulties()
        
        # 3. 종합 비교 시각화
        analyzer.create_comprehensive_comparison()
        
        # 4. 추천 보고서 생성
        analyzer.generate_route_recommendations()
        
        print("\n✅ 시뮬레이션 경로 분석 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()