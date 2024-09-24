#!/usr/bin/env python3
"""
Grid Island - 원클릭 실행 스크립트
"""

import os
from grid_island_system import GridIslandSystem

def main():
    """원클릭 실행"""
    print("🌊 Grid Island - IMU 기반 경로 최적화 시스템")
    print("="*60)

    # 시스템 실행
    system = GridIslandSystem()
    results = system.run_complete_analysis()

    if results:
        print(f"\n🎉 실행 완료!")
        print(f"📊 경로 옵션:")

        for pref, result in results.items():
            info = result['info']
            print(f"  {pref.upper():8s}: 비용={info['total_cost']:8.1f}, 세그먼트={info['segments']:3d}개")

        print(f"\n📁 결과 파일:")
        print(f"  - results/grid_island_nodes.csv")
        print(f"  - results/grid_island_edges.csv")
    else:
        print("❌ 실행 실패")

if __name__ == "__main__":
    main()