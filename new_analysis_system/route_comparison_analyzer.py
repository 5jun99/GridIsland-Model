#!/usr/bin/env python3
"""
경로 비교 분석기 - 최단거리 vs 최소난이도 구간 분석 및 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from matplotlib import font_manager, rc
import platform
from pathlib import Path

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')
else:  # Linux
    rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False

class RouteComparisonAnalyzer:
    """각 경로별 최단거리 vs 최소난이도 구간 비교 분석"""
    
    def __init__(self):
        self.routes_data = {}
        
    def load_analysis_results(self, summary_file="results/integrated_analysis_report.txt"):
        """통합 분석 결과 로드"""
        print("📊 통합 분석 결과 기반 경로별 상세 비교")
        print("=" * 60)
        
        # 경로 정보 (앞에서 분석한 결과)
        self.routes_summary = {
            'real_original': {'name': '실제 측정 원본데이터', 'avg_difficulty': 0.508, 'distance': 309, 'color': 'red'},
            'real_measured': {'name': '실제 측정 경로', 'avg_difficulty': 0.000, 'distance': 309, 'color': 'darkred'},
            'flat_bypass': {'name': '평지 우회로', 'avg_difficulty': 0.000, 'distance': 400, 'color': 'green'},
            'slope_route': {'name': '언덕길', 'avg_difficulty': 0.000, 'distance': 350, 'color': 'orange'},
            'stairs_shortcut': {'name': '계단 지름길', 'avg_difficulty': 0.187, 'distance': 280, 'color': 'purple'},
            'rough_path': {'name': '울퉁불퉁한 길', 'avg_difficulty': 0.075, 'distance': 380, 'color': 'brown'},
            'mixed_complex': {'name': '복합 경로', 'avg_difficulty': 0.064, 'distance': 450, 'color': 'pink'}
        }
        
        return self.routes_summary
    
    def create_distance_vs_difficulty_comparison(self):
        """거리 vs 난이도 비교 분석"""
        print("\n🔍 거리 vs 난이도 비교 분석")
        
        # 데이터 준비
        routes = []
        for route_id, info in self.routes_summary.items():
            routes.append({
                'route_id': route_id,
                'name': info['name'],
                'distance': info['distance'],
                'difficulty': info['avg_difficulty'],
                'color': info['color'],
                'efficiency': info['distance'] / (info['avg_difficulty'] + 0.001)  # 0으로 나누기 방지
            })
        
        df = pd.DataFrame(routes)
        
        # 최단거리 경로 찾기
        shortest_route = df.loc[df['distance'].idxmin()]
        
        # 최소난이도 경로 찾기 (0인 경우들 중에서는 거리가 짧은 것)
        min_difficulty_routes = df[df['difficulty'] == df['difficulty'].min()]
        easiest_route = min_difficulty_routes.loc[min_difficulty_routes['distance'].idxmin()]
        
        print(f"🏃‍♂️ 최단거리 경로: {shortest_route['name']} ({shortest_route['distance']}m, 난이도 {shortest_route['difficulty']:.3f})")
        print(f"😌 최소난이도 경로: {easiest_route['name']} ({easiest_route['distance']}m, 난이도 {easiest_route['difficulty']:.3f})")
        
        # 효율성 분석 (거리 대비 난이도)
        df['difficulty_per_100m'] = (df['difficulty'] * 100) / df['distance']
        best_efficiency = df.loc[df['efficiency'].idxmax()]
        print(f"⚡ 최고 효율성: {best_efficiency['name']} (효율성 지수: {best_efficiency['efficiency']:.1f})")
        
        return {
            'shortest': shortest_route.to_dict(),
            'easiest': easiest_route.to_dict(),
            'most_efficient': best_efficiency.to_dict(),
            'all_routes': df
        }
    
    def create_detailed_comparison_map(self, save_path="results/detailed_route_comparison_map.html"):
        """상세 경로 비교 지도 생성"""
        print(f"\n🗺️  상세 비교 지도 생성: {save_path}")
        
        # 중심점 (실제 GPS 좌표 범위 기반)
        center_lat = 37.620018
        center_lon = 127.058780
        
        # Folium 지도 생성
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=15,
            tiles='OpenStreetMap'
        )
        
        # 각 경로별 대략적인 GPS 좌표 생성 (시뮬레이션 기반)
        route_coordinates = {
            'real_original': [
                [37.620674, 127.057347], [37.620400, 127.058500], 
                [37.620000, 127.059500], [37.619357, 127.060591]
            ],
            'flat_bypass': [
                [37.619400, 127.057400], [37.619600, 127.058200], 
                [37.620200, 127.059800], [37.620600, 127.060500]
            ],
            'slope_route': [
                [37.619500, 127.057800], [37.620100, 127.058600], 
                [37.620500, 127.059000], [37.620700, 127.059200]
            ],
            'stairs_shortcut': [
                [37.620400, 127.057600], [37.620000, 127.058700], [37.619600, 127.059800]
            ],
            'rough_path': [
                [37.620200, 127.057300], [37.620000, 127.058000], 
                [37.619800, 127.059200], [37.619800, 127.060200]
            ],
            'mixed_complex': [
                [37.619200, 127.057500], [37.619800, 127.058300], 
                [37.620400, 127.059100], [37.620600, 127.059900], [37.620800, 127.060400]
            ]
        }
        
        # 경로별 표시
        for route_id, info in self.routes_summary.items():
            if route_id in route_coordinates:
                coords = route_coordinates[route_id]
                
                # 난이도에 따른 선 굵기 결정
                weight = 3 + (info['avg_difficulty'] * 10)  # 난이도가 높을수록 굵게
                
                # 경로 라인 그리기
                folium.PolyLine(
                    coords,
                    color=info['color'],
                    weight=weight,
                    opacity=0.8,
                    popup=f"{info['name']}<br>거리: {info['distance']}m<br>난이도: {info['avg_difficulty']:.3f}"
                ).add_to(m)
                
                # 시작점 마커
                folium.Marker(
                    coords[0],
                    popup=f"🚀 {info['name']} 시작<br>거리: {info['distance']}m",
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(m)
                
                # 끝점 마커
                folium.Marker(
                    coords[-1],
                    popup=f"🏁 {info['name']} 도착<br>난이도: {info['avg_difficulty']:.3f}",
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(m)
        
        # 특별 표시: 최단거리 vs 최소난이도
        comparison_data = self.create_distance_vs_difficulty_comparison()
        
        # 최단거리 경로 하이라이트
        shortest_id = None
        easiest_id = None
        
        for route_id, info in self.routes_summary.items():
            if info['distance'] == comparison_data['shortest']['distance']:
                shortest_id = route_id
            if info['avg_difficulty'] == comparison_data['easiest']['difficulty'] and info['distance'] == comparison_data['easiest']['distance']:
                easiest_id = route_id
        
        # 범례 및 분석 정보 추가
        legend_html = f'''
        <div style="position: fixed; 
                   bottom: 50px; left: 50px; width: 350px; height: 300px; 
                   background-color: white; border:2px solid grey; z-index:9999; 
                   font-size:12px; padding: 10px; overflow-y: auto;">
        <h4>📊 휠체어 경로 비교 분석</h4>
        
        <h5>🏃‍♂️ 최단거리</h5>
        <p><strong>{comparison_data['shortest']['name']}</strong><br>
        거리: {comparison_data['shortest']['distance']}m<br>
        난이도: {comparison_data['shortest']['difficulty']:.3f}</p>
        
        <h5>😌 최소난이도</h5>
        <p><strong>{comparison_data['easiest']['name']}</strong><br>
        거리: {comparison_data['easiest']['distance']}m<br>
        난이도: {comparison_data['easiest']['difficulty']:.3f}</p>
        
        <h5>⚡ 최고효율성</h5>
        <p><strong>{comparison_data['most_efficient']['name']}</strong><br>
        효율성: {comparison_data['most_efficient']['efficiency']:.1f}</p>
        
        <h5>📏 선 굵기</h5>
        <p>굵을수록 높은 난이도</p>
        
        <h5>🎨 색상 코드</h5>
        <p><i style="color:red">■</i> 실제 원본 (어려움)<br>
        <i style="color:green">■</i> 평지 (쉬움)<br>
        <i style="color:purple">■</i> 계단 (중간)<br>
        <i style="color:brown">■</i> 울퉁불퉁 (중간)<br></p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # 지도 저장
        m.save(save_path)
        print(f"💾 상세 비교 지도 저장 완료")
        
        return m
    
    def create_comprehensive_analysis_chart(self, save_path="results/comprehensive_route_analysis.png"):
        """종합 분석 차트 생성"""
        print(f"\n📊 종합 분석 차트 생성: {save_path}")
        
        comparison_data = self.create_distance_vs_difficulty_comparison()
        df = comparison_data['all_routes']
        
        # 4x2 서브플롯 생성
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # 1. 거리 비교
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(df)), df['distance'], color=[self.routes_summary[rid]['color'] for rid in df['route_id']])
        ax1.set_title('경로별 거리 비교', fontsize=14, fontweight='bold')
        ax1.set_ylabel('거리 (m)')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels([name.split()[0] for name in df['name']], rotation=45)
        
        # 최단거리 하이라이트
        shortest_idx = df['distance'].idxmin()
        bars[shortest_idx].set_edgecolor('red')
        bars[shortest_idx].set_linewidth(3)
        
        # 2. 난이도 비교
        ax2 = axes[0, 1]
        bars = ax2.bar(range(len(df)), df['difficulty'], color=[self.routes_summary[rid]['color'] for rid in df['route_id']])
        ax2.set_title('경로별 난이도 비교', fontsize=14, fontweight='bold')
        ax2.set_ylabel('난이도 점수')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels([name.split()[0] for name in df['name']], rotation=45)
        
        # 최소난이도 하이라이트
        easiest_idx = df['difficulty'].idxmin()
        bars[easiest_idx].set_edgecolor('blue')
        bars[easiest_idx].set_linewidth(3)
        
        # 3. 거리 vs 난이도 산점도
        ax3 = axes[0, 2]
        colors = [self.routes_summary[rid]['color'] for rid in df['route_id']]
        scatter = ax3.scatter(df['distance'], df['difficulty'], c=colors, s=100, alpha=0.7)
        ax3.set_xlabel('거리 (m)')
        ax3.set_ylabel('난이도')
        ax3.set_title('거리 vs 난이도', fontsize=14, fontweight='bold')
        
        # 각 점에 경로명 라벨
        for i, (idx, row) in enumerate(df.iterrows()):
            ax3.annotate(row['name'].split()[0], (row['distance'], row['difficulty']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. 효율성 지수
        ax4 = axes[0, 3]
        bars = ax4.bar(range(len(df)), df['efficiency'], color=[self.routes_summary[rid]['color'] for rid in df['route_id']])
        ax4.set_title('경로별 효율성 지수', fontsize=14, fontweight='bold')
        ax4.set_ylabel('효율성 (거리/난이도)')
        ax4.set_xticks(range(len(df)))
        ax4.set_xticklabels([name.split()[0] for name in df['name']], rotation=45)
        
        # 최고효율성 하이라이트
        most_efficient_idx = df['efficiency'].idxmax()
        bars[most_efficient_idx].set_edgecolor('gold')
        bars[most_efficient_idx].set_linewidth(3)
        
        # 5. 100m당 난이도
        ax5 = axes[1, 0]
        bars = ax5.bar(range(len(df)), df['difficulty_per_100m'], color=[self.routes_summary[rid]['color'] for rid in df['route_id']])
        ax5.set_title('100m당 난이도', fontsize=14, fontweight='bold')
        ax5.set_ylabel('100m당 난이도')
        ax5.set_xticks(range(len(df)))
        ax5.set_xticklabels([name.split()[0] for name in df['name']], rotation=45)
        
        # 6. 휠체어 추천 점수
        ax6 = axes[1, 1]
        # 추천 점수 = (1 - 정규화된 난이도) * (1 - 정규화된 거리)
        norm_difficulty = df['difficulty'] / df['difficulty'].max() if df['difficulty'].max() > 0 else 0
        norm_distance = (df['distance'] - df['distance'].min()) / (df['distance'].max() - df['distance'].min())
        recommendation_score = (1 - norm_difficulty) * (1 - norm_distance * 0.3)  # 거리는 30% 가중치
        
        bars = ax6.bar(range(len(df)), recommendation_score, color=[self.routes_summary[rid]['color'] for rid in df['route_id']])
        ax6.set_title('휠체어 추천 점수', fontsize=14, fontweight='bold')
        ax6.set_ylabel('추천 점수 (높을수록 좋음)')
        ax6.set_xticks(range(len(df)))
        ax6.set_xticklabels([name.split()[0] for name in df['name']], rotation=45)
        
        # 최고 추천 하이라이트
        best_recommendation_idx = recommendation_score.idxmax()
        bars[best_recommendation_idx].set_edgecolor('green')
        bars[best_recommendation_idx].set_linewidth(3)
        
        # 7. 시간 효율성 (가정: 평균 속도 1.5m/s)
        ax7 = axes[1, 2]
        estimated_time = df['distance'] / 1.5 / 60  # 분 단위
        bars = ax7.bar(range(len(df)), estimated_time, color=[self.routes_summary[rid]['color'] for rid in df['route_id']])
        ax7.set_title('예상 소요 시간', fontsize=14, fontweight='bold')
        ax7.set_ylabel('시간 (분)')
        ax7.set_xticks(range(len(df)))
        ax7.set_xticklabels([name.split()[0] for name in df['name']], rotation=45)
        
        # 8. 종합 요약 텍스트
        ax8 = axes[1, 3]
        ax8.axis('off')
        
        summary_text = f"""
휠체어 사용자 경로 추천 요약

🏃‍♂️ 최단거리
{comparison_data['shortest']['name']}
거리: {comparison_data['shortest']['distance']}m
난이도: {comparison_data['shortest']['difficulty']:.3f}

😌 최소난이도  
{comparison_data['easiest']['name']}
거리: {comparison_data['easiest']['distance']}m
난이도: {comparison_data['easiest']['difficulty']:.3f}

⚡ 최고효율성
{comparison_data['most_efficient']['name']}
효율성: {comparison_data['most_efficient']['efficiency']:.1f}

💡 추천: {df.iloc[best_recommendation_idx]['name']}
(종합 점수 최고)
        """
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 종합 분석 차트 저장 완료")
        plt.show()
        
        return fig
    
    def generate_final_recommendation_report(self, save_path="results/final_route_recommendation.txt"):
        """최종 경로 추천 보고서 생성"""
        print(f"\n📋 최종 추천 보고서 생성: {save_path}")
        
        comparison_data = self.create_distance_vs_difficulty_comparison()
        
        report = []
        report.append("🚀 휠체어 사용자를 위한 최종 경로 추천 보고서")
        report.append("=" * 60)
        report.append("📅 분석일시: 2025-10-13")
        report.append("📍 분석지역: 실제 GPS 좌표 (37.620°N, 127.059°E) 기반")
        report.append("")
        
        report.append("🎯 분석 결과 요약:")
        report.append(f"• 총 분석 경로: {len(self.routes_summary)}개")
        report.append(f"• 최단거리: {comparison_data['shortest']['name']} ({comparison_data['shortest']['distance']}m)")
        report.append(f"• 최소난이도: {comparison_data['easiest']['name']} (난이도 {comparison_data['easiest']['difficulty']:.3f})")
        report.append(f"• 최고효율성: {comparison_data['most_efficient']['name']} (효율성 {comparison_data['most_efficient']['efficiency']:.1f})")
        report.append("")
        
        report.append("🏆 상황별 추천 경로:")
        report.append("")
        
        # 상황별 추천
        df = comparison_data['all_routes']
        
        # 1. 안전 최우선
        safest = df.loc[df['difficulty'].idxmin()]
        report.append(f"🛡️  안전 최우선시:")
        report.append(f"   추천: {safest['name']}")
        report.append(f"   이유: 가장 낮은 난이도 ({safest['difficulty']:.3f})")
        report.append("")
        
        # 2. 시간 효율성 우선
        fastest = df.loc[df['distance'].idxmin()]
        report.append(f"⏰ 시간 효율성 우선:")
        report.append(f"   추천: {fastest['name']}")
        report.append(f"   이유: 가장 짧은 거리 ({fastest['distance']}m)")
        report.append("")
        
        # 3. 균형 잡힌 선택
        balanced_score = []
        for _, row in df.iterrows():
            # 정규화된 점수 (낮은 난이도와 적당한 거리)
            norm_diff = row['difficulty'] / df['difficulty'].max() if df['difficulty'].max() > 0 else 0
            norm_dist = (row['distance'] - df['distance'].min()) / (df['distance'].max() - df['distance'].min())
            balanced = (1 - norm_diff) * 0.7 + (1 - norm_dist) * 0.3
            balanced_score.append(balanced)
        
        best_balanced_idx = np.argmax(balanced_score)
        balanced_route = df.iloc[best_balanced_idx]
        
        report.append(f"⚖️  균형잡힌 선택:")
        report.append(f"   추천: {balanced_route['name']}")
        report.append(f"   이유: 안전성과 효율성의 최적 균형")
        report.append(f"   거리: {balanced_route['distance']}m, 난이도: {balanced_route['difficulty']:.3f}")
        report.append("")
        
        report.append("📊 전체 경로 상세 분석:")
        report.append("-" * 40)
        for _, row in df.iterrows():
            report.append(f"\n🛣️  {row['name']}")
            report.append(f"   거리: {row['distance']}m")
            report.append(f"   난이도: {row['difficulty']:.3f}")
            report.append(f"   효율성: {row['efficiency']:.1f}")
            report.append(f"   예상시간: {row['distance']/1.5/60:.1f}분")
            
            # 추천 등급
            if row['difficulty'] == 0:
                grade = "A+ (매우 안전)"
            elif row['difficulty'] < 0.1:
                grade = "A (안전)"
            elif row['difficulty'] < 0.2:
                grade = "B (보통)"
            elif row['difficulty'] < 0.5:
                grade = "C (주의)"
            else:
                grade = "D (위험)"
            
            report.append(f"   휠체어 적합성: {grade}")
        
        report.append("\n💡 추가 고려사항:")
        report.append("• 날씨 조건에 따른 노면 상태 변화")
        report.append("• 개별 휠체어 사용자의 신체 능력")
        report.append("• 시간대별 교통량 및 보행자 밀도")
        report.append("• 응급상황 시 접근 가능한 의료시설")
        
        report.append("\n📞 문의 및 피드백:")
        report.append("이 분석 결과에 대한 문의나 개선 사항이 있으시면")
        report.append("휠체어 접근성 개선 프로젝트팀으로 연락주시기 바랍니다.")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # 파일로 저장
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n💾 최종 추천 보고서 저장 완료")
        
        return report_text

def main():
    """메인 실행 함수"""
    analyzer = RouteComparisonAnalyzer()
    
    # 1. 분석 결과 로드
    analyzer.load_analysis_results()
    
    # 2. 거리 vs 난이도 비교
    comparison_data = analyzer.create_distance_vs_difficulty_comparison()
    
    # 3. 상세 비교 지도 생성
    analyzer.create_detailed_comparison_map()
    
    # 4. 종합 분석 차트 생성
    analyzer.create_comprehensive_analysis_chart()
    
    # 5. 최종 추천 보고서 생성
    analyzer.generate_final_recommendation_report()
    
    print("\n✅ 모든 분석 완료!")
    print("📁 생성된 파일들:")
    print("  - results/detailed_route_comparison_map.html")
    print("  - results/comprehensive_route_analysis.png")
    print("  - results/final_route_recommendation.txt")

if __name__ == "__main__":
    main()