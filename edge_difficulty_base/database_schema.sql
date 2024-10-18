-- =====================================================
-- 엣지 기반 난이도 분석 DB 스키마
-- 노드-엣지-세그먼트 구조로 설계
-- =====================================================

-- 1. 노드 테이블 (교차점, 출발/도착점)
CREATE TABLE nodes (
    node_id VARCHAR(50) PRIMARY KEY,           -- 'node_1', 'node_2' 등
    latitude DECIMAL(10, 8) NOT NULL,          -- 위도 (37.123456 형식)
    longitude DECIMAL(11, 8) NOT NULL,         -- 경도 (127.123456 형식)
    node_name VARCHAR(100),                    -- 노드 이름 (선택사항)
    node_type VARCHAR(20) DEFAULT 'waypoint', -- 'start', 'end', 'waypoint' 등
    matched_gps_index INT,                     -- 매칭된 GPS 데이터 인덱스
    match_distance DECIMAL(8, 2),             -- 매칭 거리 (미터)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_location (latitude, longitude),
    INDEX idx_node_type (node_type)
);

-- 2. 엣지 테이블 (노드 간 연결 경로)
CREATE TABLE edges (
    edge_id VARCHAR(100) PRIMARY KEY,         -- 'node_1_to_node_2' 형식
    from_node_id VARCHAR(50) NOT NULL,        -- 시작 노드
    to_node_id VARCHAR(50) NOT NULL,          -- 끝 노드
    
    -- GPS 경로 정보
    start_gps_index INT,                      -- 시작 GPS 인덱스
    end_gps_index INT,                        -- 끝 GPS 인덱스
    path_distance DECIMAL(10, 2),            -- 실제 경로 거리 (미터)
    path_duration DECIMAL(8, 2),             -- 경로 소요 시간 (초)
    
    -- 종합 난이도 정보 (세그먼트 클러스터링 결과)
    total_segments INT,                       -- 총 세그먼트 수
    difficulty_score DECIMAL(5, 3),          -- 가중 난이도 점수 (0~1)
    difficulty_level VARCHAR(20),            -- '쉬움', '보통', '어려움'
    difficulty_grade INT,                     -- 0: 쉬움, 1: 보통, 2: 어려움
    
    -- 클러스터 분포 (JSON 형태)
    cluster_distribution JSON,               -- {0: 0.3, 1: 0.5, 2: 0.2}
    avg_segment_difficulty DECIMAL(5, 3),   -- 세그먼트 평균 난이도
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (from_node_id) REFERENCES nodes(node_id),
    FOREIGN KEY (to_node_id) REFERENCES nodes(node_id),
    INDEX idx_from_node (from_node_id),
    INDEX idx_to_node (to_node_id),
    INDEX idx_difficulty_level (difficulty_level),
    INDEX idx_difficulty_score (difficulty_score)
);

-- 3. 세그먼트 테이블 (엣지의 세부 구간)
CREATE TABLE segments (
    segment_id INT PRIMARY KEY AUTO_INCREMENT,
    edge_id VARCHAR(100) NOT NULL,           -- 소속 엣지
    segment_number INT NOT NULL,             -- 엣지 내 세그먼트 순서 (1, 2, 3...)
    
    -- 시간 범위
    start_time DECIMAL(10, 2),               -- 시작 시간 (초)
    end_time DECIMAL(10, 2),                 -- 끝 시간 (초)
    duration DECIMAL(8, 2),                  -- 구간 길이 (초)
    
    -- 센서 기반 난이도 지표
    vibration_rms DECIMAL(8, 5),            -- 진동 RMS
    vibration_std DECIMAL(8, 5),            -- 진동 표준편차
    vibration_max DECIMAL(8, 5),            -- 진동 최대값
    rotation_mean DECIMAL(8, 5),            -- 회전 평균
    rotation_std DECIMAL(8, 5),             -- 회전 표준편차
    rotation_max DECIMAL(8, 5),             -- 회전 최대값
    height_change DECIMAL(8, 2),            -- 고도 변화 (미터)
    velocity_mean DECIMAL(8, 2),            -- 평균 속도
    velocity_std DECIMAL(8, 2),             -- 속도 표준편차
    
    -- 클러스터링 결과
    cluster_label INT,                       -- 클러스터 라벨 (0, 1, 2...)
    difficulty_score DECIMAL(5, 3),         -- 세그먼트 난이도 점수
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (edge_id) REFERENCES edges(edge_id),
    INDEX idx_edge_segment (edge_id, segment_number),
    INDEX idx_cluster_label (cluster_label),
    INDEX idx_difficulty_score (difficulty_score),
    UNIQUE KEY uk_edge_segment (edge_id, segment_number)
);

-- 4. 난이도 클러스터 정의 테이블
CREATE TABLE difficulty_clusters (
    cluster_id INT PRIMARY KEY,              -- 클러스터 ID (0, 1, 2...)
    cluster_name VARCHAR(50) NOT NULL,       -- '쉬움', '보통', '어려움'
    color_code VARCHAR(10),                  -- 시각화용 색상 (#green, #orange, #red)
    difficulty_range_min DECIMAL(5, 3),     -- 난이도 범위 최소값
    difficulty_range_max DECIMAL(5, 3),     -- 난이도 범위 최대값
    description TEXT,                        -- 클러스터 설명
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. GPS 경로 데이터 테이블 (선택사항 - 원본 데이터 보관용)
CREATE TABLE gps_tracks (
    track_id INT PRIMARY KEY AUTO_INCREMENT,
    edge_id VARCHAR(100),                    -- 연관된 엣지 (NULL 가능)
    gps_index INT,                           -- 원본 GPS 데이터 순서
    timestamp_sec DECIMAL(10, 2),           -- 시간 (초)
    latitude DECIMAL(10, 8),                -- 위도
    longitude DECIMAL(11, 8),               -- 경도
    height DECIMAL(8, 2),                   -- 고도 (미터)
    velocity DECIMAL(8, 2),                 -- 속도 (m/s)
    
    FOREIGN KEY (edge_id) REFERENCES edges(edge_id),
    INDEX idx_edge_gps (edge_id, gps_index),
    INDEX idx_timestamp (timestamp_sec),
    INDEX idx_location (latitude, longitude)
);

-- =====================================================
-- 초기 데이터 삽입
-- =====================================================

-- 난이도 클러스터 정의
INSERT INTO difficulty_clusters (cluster_id, cluster_name, color_code, difficulty_range_min, difficulty_range_max, description) VALUES
(0, '쉬움', '#green', 0.0, 0.33, '진동과 회전이 적고 평탄한 구간'),
(1, '보통', '#orange', 0.33, 0.66, '중간 수준의 진동과 회전이 있는 구간'),
(2, '어려움', '#red', 0.66, 1.0, '진동과 회전이 심하고 험난한 구간');

-- =====================================================
-- API 응답용 뷰 생성
-- =====================================================

-- 엣지별 난이도 정보 (API 응답용)
CREATE VIEW v_edge_difficulty AS
SELECT 
    e.edge_id,
    e.from_node_id,
    e.to_node_id,
    fn.latitude as from_lat,
    fn.longitude as from_lng,
    tn.latitude as to_lat,
    tn.longitude as to_lng,
    e.path_distance,
    e.path_duration,
    e.total_segments,
    e.difficulty_score,
    e.difficulty_level,
    e.difficulty_grade,
    e.cluster_distribution,
    e.avg_segment_difficulty
FROM edges e
JOIN nodes fn ON e.from_node_id = fn.node_id
JOIN nodes tn ON e.to_node_id = tn.node_id;

-- 세그먼트별 상세 정보 (API 응답용)
CREATE VIEW v_segment_details AS
SELECT 
    s.segment_id,
    s.edge_id,
    s.segment_number,
    s.start_time,
    s.end_time,
    s.duration,
    s.cluster_label,
    dc.cluster_name,
    dc.color_code,
    s.difficulty_score,
    s.vibration_rms,
    s.rotation_mean,
    s.height_change
FROM segments s
JOIN difficulty_clusters dc ON s.cluster_label = dc.cluster_id;

-- =====================================================
-- 인덱스 추가 (성능 최적화)
-- =====================================================

-- 복합 인덱스
CREATE INDEX idx_edge_difficulty_level_score ON edges(difficulty_level, difficulty_score);
CREATE INDEX idx_segment_edge_cluster ON segments(edge_id, cluster_label);

-- 공간 인덱스 (MySQL 8.0+에서 지원)
-- ALTER TABLE nodes ADD SPATIAL INDEX idx_spatial_location (point(longitude, latitude));