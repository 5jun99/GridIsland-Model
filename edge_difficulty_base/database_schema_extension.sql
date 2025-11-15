-- =====================================================
-- 기존 segments 테이블에 네비게이션 필드 추가
-- 기존 데이터는 유지되며, 새로운 필드만 추가됩니다.
-- =====================================================

-- 네비게이션 관련 필드 추가
ALTER TABLE segments 
ADD COLUMN start_lat DECIMAL(10, 8) NULL COMMENT '세그먼트 시작 위도',
ADD COLUMN start_lon DECIMAL(11, 8) NULL COMMENT '세그먼트 시작 경도',
ADD COLUMN end_lat DECIMAL(10, 8) NULL COMMENT '세그먼트 끝 위도',
ADD COLUMN end_lon DECIMAL(11, 8) NULL COMMENT '세그먼트 끝 경도',
ADD COLUMN distance_meters DECIMAL(8, 2) NULL COMMENT '세그먼트 거리(미터)',
ADD COLUMN bearing_degrees DECIMAL(5, 1) NULL COMMENT '방향각(북쪽=0도)',
ADD COLUMN turn_angle DECIMAL(6, 1) NULL COMMENT '회전각(-180~180도)',
ADD COLUMN navigation_instruction TEXT NULL COMMENT '경로 안내 메시지',
ADD COLUMN warning_message TEXT NULL COMMENT '주의사항 메시지',
ADD COLUMN estimated_time_sec DECIMAL(6, 1) NULL COMMENT '예상 소요시간(초)',
ADD COLUMN accessibility_level VARCHAR(50) NULL COMMENT '휠체어 접근성 수준',
ADD COLUMN is_merged BOOLEAN DEFAULT FALSE COMMENT '병합된 세그먼트 여부',
ADD COLUMN original_segment_ids TEXT NULL COMMENT '병합 전 원본 세그먼트 ID들(JSON)';

-- 성능 최적화를 위한 인덱스 추가
CREATE INDEX idx_segments_location ON segments(start_lat, start_lon);
CREATE INDEX idx_segments_difficulty_distance ON segments(difficulty_score, distance_meters);
CREATE INDEX idx_segments_merged ON segments(is_merged);

-- =====================================================
-- 네비게이션 분석용 뷰 생성
-- =====================================================

-- 경로 안내용 세그먼트 뷰 (병합된 세그먼트만)
CREATE OR REPLACE VIEW v_navigation_segments AS
SELECT 
    s.segment_id,
    s.edge_id,
    s.segment_number,
    s.start_lat,
    s.start_lon,
    s.end_lat,
    s.end_lon,
    s.distance_meters,
    s.bearing_degrees,
    s.turn_angle,
    s.navigation_instruction,
    s.warning_message,
    s.estimated_time_sec,
    s.accessibility_level,
    s.difficulty_score,
    s.cluster_label,
    dc.cluster_name,
    dc.color_code,
    s.is_merged,
    s.original_segment_ids
FROM segments s
JOIN difficulty_clusters dc ON s.cluster_label = dc.cluster_id
WHERE s.navigation_instruction IS NOT NULL
ORDER BY s.edge_id, s.segment_number;

-- 경로별 요약 정보 뷰
CREATE OR REPLACE VIEW v_route_summary AS
SELECT 
    s.edge_id,
    COUNT(*) as total_navigation_segments,
    SUM(s.distance_meters) as total_distance,
    SUM(s.estimated_time_sec) as total_estimated_time,
    AVG(s.difficulty_score) as avg_difficulty,
    MAX(s.difficulty_score) as max_difficulty,
    COUNT(CASE WHEN s.difficulty_score > 0.7 THEN 1 END) as high_difficulty_segments,
    GROUP_CONCAT(DISTINCT s.accessibility_level) as accessibility_levels
FROM segments s
WHERE s.navigation_instruction IS NOT NULL
GROUP BY s.edge_id;

COMMIT;