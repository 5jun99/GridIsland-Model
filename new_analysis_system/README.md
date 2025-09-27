# 🔍 Grid Island - 새로운 분석 시스템

**Test 데이터 특성 추출 및 클러스터링 분석**

## 🎯 시스템 개요

이 시스템은 IMU 센서 데이터에서 의미있는 특성을 추출하고 클러스터링을 통해 패턴을 발견하는 기초 분석 도구입니다.

### 주요 기능
- ✅ **센서 데이터 로딩**: 6축 IMU 센서 데이터 통합 처리
- ✅ **고급 특성 추출**: 시간/주파수/모션 도메인에서 55+ 특성 추출
- ✅ **클러스터링 분석**: K-Means, GMM, DBSCAN 등 다양한 방법 지원
- ✅ **최적화**: 자동으로 최적 클러스터 수 탐색
- ✅ **시각화**: t-SNE, PCA 기반 결과 시각화

## 🚀 빠른 시작

### 1. 원클릭 분석
```bash
python analyze_test_data.py
```

### 2. 단계별 실행

#### 특성 추출만
```bash
python feature_extractor.py
```

#### 클러스터링만 (특성 파일이 있을 때)
```bash
python clustering_analyzer.py
```

## 📁 프로젝트 구조

```
new_analysis_system/
├── analyze_test_data.py        # 🚀 메인 분석 파이프라인
├── feature_extractor.py        # 🔍 특성 추출기
├── clustering_analyzer.py      # 🎯 클러스터링 분석기
├── requirements.txt            # 📦 필수 라이브러리
├── README.md                   # 📖 사용 설명서
├── utils/                      # 🛠️ 유틸리티
│   ├── __init__.py
│   └── data_loader.py          # 센서 데이터 로더
├── data/                       # 📊 입력 데이터
│   └── test 2025-09-22 18-30-21/
└── results/                    # 📈 분석 결과
    ├── extracted_features.csv
    ├── clustered_features.csv
    ├── cluster_characteristics.csv
    ├── cluster_optimization.csv
    └── clustering_visualization.png
```

## 🔧 특성 추출 상세

### 시간 도메인 특성 (Time Domain)
- **기본 통계**: 평균, 표준편차, 분산, 최소/최대값
- **분포 특성**: 중앙값, 분위수, 왜도, 첨도
- **신호 품질**: RMS, Zero Crossing Rate
- **변화량**: 평균 변화율, 변화율 표준편차

### 주파수 도메인 특성 (Frequency Domain)
- **스펙트럼 특성**: Spectral Centroid, Rolloff, Bandwidth
- **주파수 대역 에너지**: 저주파(0-2Hz), 중주파(2-8Hz), 고주파(8-20Hz)
- **도미넌트 주파수**: 가장 강한 주파수 성분

### 모션 특성 (Motion Features)
- **가속도 벡터**: 3축 가속도 합성 벡터의 특성
- **자이로스코프**: 3축 각속도 특성
- **Jerk**: 가속도 변화율 (급격한 움직임 감지)
- **활동 강도**: 전체적인 움직임 활성도

## 🎯 클러스터링 분석

### 지원 알고리즘
- **K-Means**: 효율적인 구형 클러스터 탐지
- **Gaussian Mixture Model**: 확률적 클러스터링
- **DBSCAN**: 밀도 기반 이상치 탐지

### 평가 지표
- **Silhouette Score**: 클러스터 분리도 (-1 ~ 1, 높을수록 좋음)
- **Calinski-Harabasz Index**: 클러스터 내 응집도 vs 클러스터 간 분산
- **Davies-Bouldin Index**: 클러스터 내 분산 vs 클러스터 간 거리 (낮을수록 좋음)

### 최적화 과정
1. K=2부터 10까지 자동 탐색
2. 각 K값에 대해 평가 지표 계산
3. Silhouette Score가 최대인 K값 추천
4. 결과 시각화로 검증

## 📊 결과 해석

### 출력 파일들

#### `extracted_features.csv`
- 각 윈도우별 추출된 모든 특성
- 컬럼: window_id, start_idx, end_idx + 55개 특성

#### `clustered_features.csv`
- 특성 데이터 + 클러스터 라벨
- 시간에 따른 클러스터 변화 추적 가능

#### `cluster_characteristics.csv`
- 클러스터별 특성 요약 통계
- 각 클러스터의 특징적인 특성값들

#### `cluster_optimization.csv`
- K값별 평가 지표 변화
- 최적 클러스터 수 선택 근거

#### `clustering_visualization.png`
- t-SNE 2D 시각화
- PCA 결과 (있는 경우)
- 클러스터 분포 히스토그램
- 시간에 따른 클러스터 변화

## 🔧 설정 옵션

### FeatureExtractor 설정
```python
extractor = FeatureExtractor(
    window_size=200,      # 윈도우 크기 (샘플 수)
    overlap_ratio=0.75    # 오버랩 비율 (0.0 ~ 1.0)
)
```

### ClusteringAnalyzer 설정
```python
# 클러스터 수 탐색 범위
analyzer.find_optimal_clusters(max_k=10, method='kmeans')

# 특정 클러스터 수로 분석
labels = analyzer.perform_clustering(n_clusters=5, method='kmeans')
```

## 🎯 사용 시나리오

### 1. 기본 패턴 발견
```bash
# 기본 설정으로 실행
python analyze_test_data.py
```

### 2. 세밀한 분석 (작은 윈도우)
```python
# feature_extractor.py 수정
extractor = FeatureExtractor(window_size=100, overlap_ratio=0.8)
```

### 3. 빠른 분석 (큰 윈도우)
```python
# feature_extractor.py 수정
extractor = FeatureExtractor(window_size=400, overlap_ratio=0.5)
```

### 4. 이상치 탐지
```python
# clustering_analyzer.py에서 DBSCAN 사용
labels = analyzer.perform_clustering(method='dbscan')
```

## 📈 성능 정보

- **처리 속도**: 7,995 샘플 → ~200 윈도우 (수 초)
- **메모리 사용량**: < 50MB
- **특성 수**: 55개 (확장 가능)
- **클러스터링**: K=2~10 자동 최적화

## 🔍 문제 해결

### 자주 발생하는 문제

1. **모듈 없음 오류**
   ```bash
   pip install -r requirements.txt
   ```

2. **시각화 오류 (GUI 환경이 없는 경우)**
   ```python
   # analyze_test_data.py에서 matplotlib backend 변경됨
   matplotlib.use('Agg')  # 파일로만 저장
   ```

3. **메모리 부족**
   ```python
   # 윈도우 크기 증가 또는 오버랩 감소
   extractor = FeatureExtractor(window_size=400, overlap_ratio=0.5)
   ```

## 💡 다음 단계

이 기초 분석 결과를 바탕으로:

1. **의미있는 클러스터 해석**: 각 클러스터가 어떤 활동/상황을 나타내는지 분석
2. **라벨링**: 클러스터를 실제 활동 유형으로 매핑
3. **모델 학습**: 분류 모델 훈련을 위한 라벨 데이터로 활용
4. **그래프 구성**: 클러스터 정보를 노드 특성으로 활용

🎉 **새로운 분석 시스템으로 데이터의 숨겨진 패턴을 발견하세요!**