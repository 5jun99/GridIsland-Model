#!/usr/bin/env python3
"""
캠퍼스 노드들을 지도에 시각화
"""

import folium
import pandas as pd

def create_node_map():
    """18개 노드와 측정된 엣지들을 지도에 표시"""
    
    # 노드 좌표 로드
    nodes = []
    with open('node_coord.txt', 'r') as f:
        for i, line in enumerate(f):
            if line.strip():
                lat, lon = map(float, line.strip().split(', '))
                nodes.append({
                    'node_id': i + 1,  # 1부터 시작
                    'latitude': lat,
                    'longitude': lon
                })
    
    # 측정된 엣지들 (18개 노드 매핑 결과에서)
    edges = [
        # 양방향 엣지 (실제 세션에서 양방향 측정됨)
        (8, 9, '양방향', '#00FF00'),   # 01세션(8→9), 02세션(9→8)
        (8, 10, '양방향', '#00FF00'),  # 03세션(8→10), 04세션(10→8)
        (2, 16, '양방향', '#00FF00'),  # 26세션(2→16), 25세션(16→2)
        (15, 16, '양방향', '#00FF00'), # 30세션(15→16), 29세션(16→15)
        (1, 16, '양방향', '#00FF00'),  # 40세션(1→16), 39세션(16→1)
        (4, 18, '양방향', '#00FF00'),  # 09세션(4→18), 20세션(18→4)
        
        # 단방향 엣지 (한 방향만 측정됨)
        (7, 6, '단방향', '#FF0000'),   # 07,08세션(7→6)
        (7, 8, '단방향', '#FF0000'),   # 05세션(7→8)
        (9, 7, '단방향', '#FF0000'),   # 06세션(9→7)
        (8, 12, '단방향', '#FF0000'),  # 11세션(8→12)
        (4, 5, '단방향', '#FF0000'),   # 10세션(4→5)
        (4, 7, '단방향', '#FF0000'),   # 12세션(4→7)
        (4, 13, '단방향', '#FF0000'),  # 19세션(4→13)
        (4, 3, '단방향', '#FF0000'),   # 16세션(4→3)
        (11, 4, '단방향', '#FF0000'),  # 13세션(11→4)
        (11, 3, '단방향', '#FF0000'),  # 14세션(11→3)
        (12, 3, '단방향', '#FF0000'),  # 15세션(12→3)
        (3, 2, '단방향', '#FF0000'),   # 23세션(3→2)
        (2, 12, '단방향', '#FF0000'),  # 24세션(2→12)
        (17, 16, '단방향', '#FF0000'), # 38세션(17→16)
        (13, 18, '단방향', '#FF0000'), # 32세션(13→18)
        (5, 18, '단방향', '#FF0000'),  # 31세션(5→18)
    ]
    
    # 지도 중심 계산 (광운대학교 캠퍼스)
    center_lat = sum(node['latitude'] for node in nodes) / len(nodes)
    center_lon = sum(node['longitude'] for node in nodes) / len(nodes)
    
    # 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles='OpenStreetMap'
    )
    
    # 노드 마커 추가
    for node in nodes:
        # 해당 노드를 사용하는 엣지 수 계산
        edge_count = 0
        
        for start, end, edge_type, color in edges:
            if start == node['node_id'] or end == node['node_id']:
                edge_count += 1
        
        # 노드 크기 조정 (엣지 수에 따라)
        radius = max(8, edge_count * 2)
        
        # 노드 색상 (연결된 엣지 수에 따라)
        if edge_count >= 4:
            node_color = '#FF6B6B'  # 빨간색 (많이 연결됨)
        elif edge_count >= 2:
            node_color = '#4ECDC4'  # 청록색 (보통)
        else:
            node_color = '#45B7D1'  # 파란색 (적게 연결됨)
        
        # 노드 마커 생성
        folium.CircleMarker(
            location=[node['latitude'], node['longitude']],
            radius=radius,
            popup=f'''
            <div style="font-family: Arial; font-size: 12px;">
                <b>노드 {node['node_id']}</b><br>
                위도: {node['latitude']:.6f}<br>
                경도: {node['longitude']:.6f}<br>
                연결된 엣지: {edge_count}개
            </div>
            ''',
            tooltip=f'노드 {node["node_id"]}',
            color='black',
            weight=2,
            fillColor=node_color,
            fillOpacity=0.8
        ).add_to(m)
        
        # 노드 번호 라벨 추가
        folium.Marker(
            location=[node['latitude'], node['longitude']],
            icon=folium.DivIcon(
                html=f'''
                <div style="
                    font-size: 12px; 
                    font-weight: bold; 
                    text-align: center;
                    color: white;
                    text-shadow: 1px 1px 1px black;
                    margin-top: -5px;
                ">{node['node_id']}</div>
                ''',
                icon_size=(20, 20),
                icon_anchor=(10, 10)
            )
        ).add_to(m)
    
    # 엣지 라인 추가
    for start_id, end_id, edge_type, color in edges:
        # 노드 ID는 1부터 시작하므로 인덱스는 -1
        start_node = nodes[start_id - 1]
        end_node = nodes[end_id - 1]
        
        # 라인 스타일
        if edge_type == '양방향':
            weight = 4
            opacity = 0.8
        else:
            weight = 3
            opacity = 0.6
        
        # 엣지 라인
        folium.PolyLine(
            locations=[
                [start_node['latitude'], start_node['longitude']],
                [end_node['latitude'], end_node['longitude']]
            ],
            color=color,
            weight=weight,
            opacity=opacity,
            popup=f'엣지: {start_id}→{end_id} ({edge_type})'
        ).add_to(m)
    
    # 범례 추가
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 220px; height: 140px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
    <b>🗺️ 캠퍼스 노드 네트워크</b><br><br>
    <b>노드 연결도:</b><br>
    <span style="color:#FF6B6B">●</span> 고연결 (4+ 엣지)<br>
    <span style="color:#4ECDC4">●</span> 중연결 (2-3 엣지)<br>
    <span style="color:#45B7D1">●</span> 저연결 (1 엣지)<br><br>
    <b>엣지 유형:</b><br>
    <span style="color:#00FF00; font-weight: bold;">━━</span> 양방향 측정<br>
    <span style="color:#FF0000; font-weight: bold;">━━</span> 단방향 측정
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 지도 저장
    output_file = 'campus_node_network.html'
    m.save(output_file)
    
    print(f"🗺️  캠퍼스 노드 네트워크 지도 생성 완료: {output_file}")
    print(f"📍 총 {len(nodes)}개 노드, {len(edges)}개 엣지")
    print(f"🌐 브라우저에서 {output_file} 파일을 열어서 확인하세요!")
    
    return output_file

if __name__ == "__main__":
    try:
        create_node_map()
    except ImportError:
        print("❌ folium 라이브러리가 필요합니다. 설치: pip install folium")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")