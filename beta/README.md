# 🌊 Grid Island - 최종 IMU 경로 최적화 시스템

**IMU 센서 데이터 기반 그래프 경로 최적화의 완전한 솔루션**

## 🎯 시스템 개요

Grid Island는 IMU 센서 데이터를 분석하여 이동 경로의 난이도를 추정하고, 그래프 기반 최적 경로를 제공하는 완전한 엔드투엔드 시스템입니다.

### 핵심 기능
- ✅ **실시간 IMU 분석**: 6축 센서 데이터 → 4단계 난이도 분류
- ✅ **그래프 경로 최적화**: 다익스트라 알고리즘 기반 최적 경로 탐색
- ✅ **3가지 선호도**: Fastest/Balanced/Safest 경로 제공
- ✅ **원클릭 솔루션**: 완전 자동화된 분석 파이프라인

## 🚀 빠른 시작

### 1. 간단한 실행
```bash
python grid_island_system.py
```

### 2. 다른 데이터로 실행
```bash
python -c "
from grid_island_system import GridIslandSystem
system = GridIslandSystem()
system.run_complete_analysis('your_data_directory')
"
```

### 3. 단계별 실행
```python
from grid_island_system import GridIslandSystem

# 시스템 초기화
system = GridIslandSystem()
system.setup_system()

# 데이터 처리
system.process_imu_data('test 2025-09-22 18-30-21')

# 경로 탐색
path, info = system.find_optimal_path(preference='balanced')
```

## 📁 파일 구조

```
final_production_system/
├── grid_island_system.py          # 🎯 메인 통합 시스템
├── production_pipeline.py         # 🔧 핵심 IMU 분석 엔진
├── final_improved_model.py        # 🧠 최적화된 ML 모델 (F1: 64%)
├── final_graph_system.py          # 🕸️ 그래프 경로 탐색 시스템
├── data/raw/                       # 📊 학습 데이터
├── models/                         # 🤖 학습된 모델 파일
├── utils/                          # 🛠️ 유틸리티 함수
├── test 2025-09-22 18-30-21/       # 📱 테스트 데이터
└── results/                        # 📈 분석 결과
```

## 🎛️ 시스템 아키텍처

```
IMU 센서 데이터 (ax,ay,az,gx,gy,gz)
    ↓ 전처리 (50Hz, 4초 윈도우, 중력보정)
55차원 특성 추출 (시간/주파수/자세)
    ↓ 최적화된 RandomForest (300 트리)
4단계 난이도 분류 (0=평지, 1=경사, 2=계단, 3=극한)
    ↓ 연속성 보정 (스무딩)
그래프 노드/엣지 생성
    ↓ 다익스트라 알고리즘
3가지 최적 경로 (Fastest/Balanced/Safest)
```

## 📊 성능 지표

- **모델 정확도**: F1 Score 0.64 (64%)
- **처리 속도**: 7,995 샘플 → 114 노드 (실시간)
- **그래프 규모**: ~100-200 노드, ~500-1000 엣지
- **메모리 사용량**: < 100MB (경량)

## 🎯 사용 시나리오

### 1. 실시간 경로 추천
```python
system = GridIslandSystem()
system.setup_system()

# 새 IMU 데이터가 들어올 때마다
results = system.process_imu_data(new_data_dir)
path, info = system.find_optimal_path(preference='safest')
```

### 2. 배치 분석
```python
data_dirs = ['route_1', 'route_2', 'route_3']

for data_dir in data_dirs:
    system.run_complete_analysis(data_dir)
    # results/에 각 경로별 분석 결과 저장
```

### 3. 경로 비교 분석
```python
# 동일 경로의 3가지 옵션 비교
results = system.run_complete_analysis()

for pref, result in results.items():
    print(f"{pref}: 비용={result['info']['total_cost']:.1f}")
```

## ⚙️ 설정 옵션

### 윈도우 설정
```python
# production_pipeline.py에서 수정
sampling_rate = 50          # 50Hz IMU
window_seconds = 4.0        # 4초 윈도우
overlap_ratio = 0.75        # 75% 오버랩
```

### 경로 선호도 조정
```python
# grid_island_system.py에서 수정
preferences = {
    'custom': {
        'distance_weight': 1.2,    # 거리 중요도
        'difficulty_weight': 0.8   # 난이도 중요도
    }
}
```

### 난이도 매핑 변경
```python
# production_pipeline.py에서 수정
difficulty_map = {
    0: "새로운 평지 정의",
    1: "새로운 경사 정의",
    2: "새로운 계단 정의",
    3: "새로운 극한 정의"
}
```

## 📈 결과 해석

### 노드 데이터 (`grid_island_nodes.csv`)
- `node_id`: 노드 고유 ID
- `difficulty`: 난이도 레벨 (0-3)
- `confidence`: 예측 신뢰도 (0-1)
- `base_cost`: 기본 이동 비용

### 엣지 데이터 (`grid_island_edges.csv`)
- `edge_id`: 엣지 고유 ID
- `from_node`, `to_node`: 연결된 노드들
- `cost`: 이동 비용 (선호도별 조정됨)
- `edge_type`: 연결 유형 (sequential/jump_2/jump_3/jump_5)

### 경로 정보
- `total_cost`: 총 경로 비용
- `segments`: 경로상 노드 수
- `difficulty_distribution`: 난이도별 구간 분포

## 🔧 문제 해결

### 자주 발생하는 문제

1. **모델 파일 없음**
   ```
   해결: 첫 실행시 자동으로 모델을 학습합니다 (1-2분 소요)
   ```

2. **메모리 부족**
   ```
   해결: window_seconds를 3.0으로 줄이거나 overlap_ratio를 0.5로 조정
   ```

3. **처리 속도 느림**
   ```
   해결:
   - n_estimators를 200으로 감소
   - 특성 수를 줄이기 (고급 특성 제거)
   ```

4. **낮은 예측 신뢰도**
   ```
   해결:
   - 더 긴 윈도우 사용 (6초)
   - 캘리브레이션 데이터로 개인화
   ```

## 🎁 주요 특징

### ✅ **Production Ready**
- 완전한 오류 처리 및 로깅
- 견고한 데이터 검증
- 메모리 효율적 처리

### ✅ **확장 가능**
- 새로운 선호도 모드 추가 가능
- 다른 그래프 알고리즘 적용 가능
- 개인화 학습 시스템 확장 가능

### ✅ **해석 가능**
- 각 예측에 대한 신뢰도 제공
- 특성 중요도 분석 가능
- 명확한 난이도 정의

## 🚀 성능 최적화 팁

1. **대용량 데이터**: 배치 처리 활용
2. **실시간 처리**: 스트리밍 모드로 윈도우별 처리
3. **정확도 향상**: 캘리브레이션 데이터 수집
4. **속도 향상**: GPU 사용 (CUDA 지원)

---

## 💡 핵심 아이디어

**"센서 데이터 → 난이도 → 그래프 → 최적 경로"**

단순하지만 강력한 파이프라인으로 실제 이동 경로의 최적화를 실현합니다.

🎉 **Grid Island와 함께 더 스마트한 경로를 경험하세요!**