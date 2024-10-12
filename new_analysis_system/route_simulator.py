#!/usr/bin/env python3
"""
경로 시뮬레이터 - 다양한 난이도의 경로 데이터 생성
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

class RouteSimulator:
    """다양한 난이도의 경로 시뮬레이션 데이터 생성기"""
    
    def __init__(self, base_coords=(37.620018, 127.058780)):  # 실제 GPS 데이터 중심점
        self.base_lat, self.base_lng = base_coords
        self.sampling_rate = 50  # 50Hz
        
        # 실제 GPS 데이터 범위 (기존 Sss 데이터 분석 결과)
        self.real_lat_range = (37.619304, 37.620806)
        self.real_lng_range = (127.057268, 127.060672)
        self.real_height_range = (26.1, 37.5)
        
    def generate_route_network(self):
        """실제 GPS 위치 기반 여러 경로 옵션 생성"""
        routes = {
            'real_measured': {
                'name': '실제 측정 경로',
                'difficulty_type': 'real',
                'distance': 309,  # 실제 측정된 거리
                'duration': 402,  # 실제 측정 시간 (초)
                'waypoints': [
                    (37.620674, 127.057347),   # 실제 시작점
                    (37.620400, 127.058500),   # 중간점 1
                    (37.620000, 127.059500),   # 중간점 2
                    (37.619357, 127.060591),   # 실제 끝점
                ]
            },
            'flat_bypass': {
                'name': '평지 우회로',
                'difficulty_type': 'flat',
                'distance': 400,
                'duration': 480,  # 8분
                'waypoints': [
                    (37.619400, 127.057400),   # 시작점
                    (37.619600, 127.058200),   # 평지 경유점 1
                    (37.620200, 127.059800),   # 평지 경유점 2
                    (37.620600, 127.060500),   # 도착점
                ]
            },
            'slope_route': {
                'name': '언덕길',
                'difficulty_type': 'slope',
                'distance': 350,
                'duration': 420,  # 7분
                'waypoints': [
                    (37.619500, 127.057800),   # 시작점
                    (37.620100, 127.058600),   # 언덕 시작
                    (37.620500, 127.059000),   # 언덕 정상
                    (37.620700, 127.059200),   # 도착점
                ]
            },
            'stairs_shortcut': {
                'name': '계단 지름길',
                'difficulty_type': 'stairs',
                'distance': 280,
                'duration': 300,  # 5분
                'waypoints': [
                    (37.620400, 127.057600),   # 시작점
                    (37.620000, 127.058700),   # 계단 구간
                    (37.619600, 127.059800),   # 도착점
                ]
            },
            'rough_path': {
                'name': '울퉁불퉁한 길',
                'difficulty_type': 'rough',
                'distance': 380,
                'duration': 540,  # 9분
                'waypoints': [
                    (37.620200, 127.057300),   # 시작점
                    (37.620000, 127.058000),   # 울퉁불퉁 구간 1
                    (37.619800, 127.059200),   # 울퉁불퉁 구간 2
                    (37.619800, 127.060200),   # 도착점
                ]
            },
            'mixed_complex': {
                'name': '복합 경로',
                'difficulty_type': 'mixed',
                'distance': 450,
                'duration': 600,  # 10분
                'waypoints': [
                    (37.619200, 127.057500),   # 시작점
                    (37.619800, 127.058300),   # 평지 구간
                    (37.620400, 127.059100),   # 언덕 구간
                    (37.620600, 127.059900),   # 울퉁불퉁 구간
                    (37.620800, 127.060400),   # 도착점
                ]
            }
        }
        return routes
    
    def simulate_flat_terrain(self, duration, sampling_rate=50):
        """평지 경로 센서 데이터 시뮬레이션"""
        samples = int(duration * sampling_rate)
        time_points = np.linspace(0, duration, samples)
        
        # 가속도 데이터 (매우 안정적)
        acc_x = np.random.normal(0, 0.1, samples)  # 좌우 흔들림 최소
        acc_y = np.random.normal(0, 0.1, samples)  # 전후 흔들림 최소  
        acc_z = np.random.normal(9.8, 0.2, samples)  # 중력 + 약간 진동
        
        # 자이로스코프 데이터 (회전 최소)
        gyro_x = np.random.normal(0, 0.05, samples)
        gyro_y = np.random.normal(0, 0.05, samples)
        gyro_z = np.random.normal(0, 0.05, samples)
        
        return {
            'time': time_points,
            'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z,
            'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z
        }
    
    def simulate_slope_terrain(self, duration, sampling_rate=50):
        """경사 경로 센서 데이터 시뮬레이션"""
        samples = int(duration * sampling_rate)
        time_points = np.linspace(0, duration, samples)
        
        # 경사 프로필 (오르막 → 평지 → 내리막)
        slope_profile = np.concatenate([
            np.linspace(0, 15, samples//3),      # 오르막 (15도)
            np.full(samples//3, 15),             # 유지
            np.linspace(15, 0, samples//3)       # 내리막
        ])
        
        # 경사에 따른 가속도 변화
        acc_x = np.random.normal(0, 0.3, samples)
        acc_y = np.sin(np.radians(slope_profile)) * 2 + np.random.normal(0, 0.4, samples)  # 전후 기울기
        acc_z = np.cos(np.radians(slope_profile)) * 9.8 + np.random.normal(0, 0.5, samples)
        
        # 자이로스코프 (기울기 변화)
        gyro_x = np.random.normal(0, 0.2, samples)
        gyro_y = np.gradient(slope_profile) * 0.1 + np.random.normal(0, 0.15, samples)  # 기울기 변화
        gyro_z = np.random.normal(0, 0.1, samples)
        
        return {
            'time': time_points,
            'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z,
            'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z
        }
    
    def simulate_stairs_terrain(self, duration, sampling_rate=50):
        """계단 경로 센서 데이터 시뮬레이션"""
        samples = int(duration * sampling_rate)
        time_points = np.linspace(0, duration, samples)
        
        # 계단 충격 패턴 (주기적인 큰 충격)
        step_frequency = 1.5  # 1.5초마다 계단
        step_impacts = np.sin(2 * np.pi * step_frequency * time_points) * 5
        step_impacts = np.where(step_impacts > 3, step_impacts, 0)  # 임계값 이상만
        
        # 가속도 데이터 (큰 충격과 진동)
        acc_x = np.random.normal(0, 1.0, samples) + step_impacts * 0.3
        acc_y = np.random.normal(2, 1.5, samples) + step_impacts * 0.5  # 앞으로 기울기
        acc_z = np.random.normal(9.8, 2.0, samples) + step_impacts
        
        # 자이로스코프 (불규칙한 회전)
        gyro_x = np.random.normal(0, 0.8, samples) + step_impacts * 0.2
        gyro_y = np.random.normal(0, 0.6, samples) + step_impacts * 0.3
        gyro_z = np.random.normal(0, 0.4, samples)
        
        return {
            'time': time_points,
            'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z,
            'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z
        }
    
    def simulate_rough_terrain(self, duration, sampling_rate=50):
        """울퉁불퉁한 길 센서 데이터 시뮬레이션"""
        samples = int(duration * sampling_rate)
        time_points = np.linspace(0, duration, samples)
        
        # 불규칙한 진동 패턴
        rough_pattern = np.random.normal(0, 1, samples)
        for i in range(5, 15):  # 다양한 주파수 성분
            rough_pattern += np.sin(2 * np.pi * i/10 * time_points) * np.random.uniform(0.2, 0.8)
        
        # 가속도 데이터 (지속적인 진동)
        acc_x = np.random.normal(0, 0.8, samples) + rough_pattern * 0.4
        acc_y = np.random.normal(0, 0.8, samples) + rough_pattern * 0.3
        acc_z = np.random.normal(9.8, 1.2, samples) + np.abs(rough_pattern)
        
        # 자이로스코프 (지속적인 흔들림)
        gyro_x = np.random.normal(0, 0.5, samples) + rough_pattern * 0.1
        gyro_y = np.random.normal(0, 0.5, samples) + rough_pattern * 0.1
        gyro_z = np.random.normal(0, 0.3, samples) + rough_pattern * 0.05
        
        return {
            'time': time_points,
            'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z,
            'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z
        }
    
    def simulate_mixed_realistic(self, duration, sampling_rate=50):
        """실제 측정 데이터와 유사한 혼합 패턴 시뮬레이션"""
        samples = int(duration * sampling_rate)
        time_points = np.linspace(0, duration, samples)
        
        # 실제 데이터와 유사한 변동 패턴 (중간 정도 난이도)
        base_variation = np.sin(2 * np.pi * 0.1 * time_points) * 0.5  # 저주파 변화
        mid_variation = np.sin(2 * np.pi * 0.5 * time_points) * 0.3   # 중주파 변화
        
        # 가속도 데이터 (실제 측정값과 유사한 범위)
        acc_x = np.random.normal(0, 0.4, samples) + base_variation
        acc_y = np.random.normal(0, 0.4, samples) + mid_variation
        acc_z = np.random.normal(9.8, 0.8, samples) + np.abs(base_variation)
        
        # 자이로스코프 (보통 수준의 회전)
        gyro_x = np.random.normal(0, 0.2, samples) + base_variation * 0.1
        gyro_y = np.random.normal(0, 0.2, samples) + mid_variation * 0.1
        gyro_z = np.random.normal(0, 0.15, samples)
        
        return {
            'time': time_points,
            'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z,
            'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z
        }
    
    def generate_gps_trajectory(self, waypoints, duration, sampling_rate=1):
        """GPS 궤적 생성"""
        gps_samples = int(duration * sampling_rate)
        
        # waypoints 사이를 보간
        total_points = len(waypoints)
        t = np.linspace(0, total_points-1, gps_samples)
        
        lats = np.interp(t, range(total_points), [wp[0] for wp in waypoints])
        lngs = np.interp(t, range(total_points), [wp[1] for wp in waypoints])
        
        # 고도 시뮬레이션 (실제 GPS 데이터 범위 기반)
        if hasattr(self, 'route_type'):
            if self.route_type == 'real':
                # 실제 측정값과 유사한 패턴
                base_heights = [35.0, 32.5, 30.0, 28.5]
            elif self.route_type == 'flat':
                base_heights = [32.0 + i*0.5 for i in range(total_points)]
            elif self.route_type == 'slope':
                base_heights = [28.0 + i*4.0 for i in range(total_points)]
            elif self.route_type == 'stairs':
                base_heights = [36.0 + i*6.0 for i in range(total_points)]
            elif self.route_type == 'rough':
                base_heights = [33.0 + np.random.uniform(-2, 2) for i in range(total_points)]
            else:  # mixed
                base_heights = [29.0, 33.0, 37.0, 35.0, 31.0][:total_points]
        else:
            base_heights = [32.0 + i*2 for i in range(total_points)]
        
        heights = np.interp(t, range(total_points), base_heights)
        # 실제 고도 범위로 제한
        heights = np.clip(heights, self.real_height_range[0], self.real_height_range[1])
        
        # 속도 계산 (거리 변화량 기반)
        velocities = []
        for i in range(len(lats)):
            if i == 0:
                velocities.append(1.5)  # 초기 속도
            else:
                # 두 점 사이 거리 계산 (근사)
                dlat = (lats[i] - lats[i-1]) * 111320  # 위도 1도 ≈ 111km
                dlng = (lngs[i] - lngs[i-1]) * 111320 * np.cos(np.radians(lats[i]))
                distance = np.sqrt(dlat**2 + dlng**2)
                velocity = distance * sampling_rate  # m/s
                velocities.append(max(0.5, min(3.0, velocity)))  # 0.5-3.0 m/s 제한
        
        time_points = np.linspace(0, duration, gps_samples)
        
        return {
            'time': time_points,
            'latitude': lats,
            'longitude': lngs, 
            'height': heights,
            'velocity': velocities,
            'direction': np.full(gps_samples, 0),  # 방향 (단순화)
            'h_accuracy': np.full(gps_samples, 3.0),  # 수평 정확도
            'v_accuracy': np.full(gps_samples, 5.0)   # 수직 정확도
        }
    
    def create_route_data(self, route_info):
        """단일 경로의 완전한 데이터 생성"""
        difficulty_type = route_info['difficulty_type']
        duration = route_info['duration']
        waypoints = route_info['waypoints']
        
        # route_type 설정 (고도 생성에 사용)
        self.route_type = difficulty_type
        
        print(f"🛣️  {route_info['name']} 데이터 생성 중...")
        
        # 센서 데이터 생성 (50Hz)
        if difficulty_type == 'real':
            # 실제 측정 데이터와 유사한 패턴
            sensor_data = self.simulate_mixed_realistic(duration)
        elif difficulty_type == 'flat':
            sensor_data = self.simulate_flat_terrain(duration)
        elif difficulty_type == 'slope':
            sensor_data = self.simulate_slope_terrain(duration)
        elif difficulty_type == 'stairs':
            sensor_data = self.simulate_stairs_terrain(duration)
        elif difficulty_type == 'rough':
            sensor_data = self.simulate_rough_terrain(duration)
        elif difficulty_type == 'mixed':
            # 구간별로 다른 시뮬레이션 적용
            quarter = duration // 4
            flat_data = self.simulate_flat_terrain(quarter)
            slope_data = self.simulate_slope_terrain(quarter)
            rough_data = self.simulate_rough_terrain(quarter)
            stairs_data = self.simulate_stairs_terrain(quarter)
            
            # 데이터 연결
            sensor_data = {}
            for key in flat_data.keys():
                if key == 'time':
                    sensor_data[key] = np.concatenate([
                        flat_data[key],
                        slope_data[key] + quarter,
                        rough_data[key] + quarter*2,
                        stairs_data[key] + quarter*3
                    ])
                else:
                    sensor_data[key] = np.concatenate([
                        flat_data[key], slope_data[key], 
                        rough_data[key], stairs_data[key]
                    ])
        else:
            sensor_data = self.simulate_rough_terrain(duration)
        
        # GPS 데이터 생성 (1Hz)
        gps_data = self.generate_gps_trajectory(waypoints, duration, sampling_rate=1)
        
        return {
            'route_info': route_info,
            'sensor_data': sensor_data,
            'gps_data': gps_data
        }
    
    def save_route_data(self, route_data, base_dir="data/simulated_routes"):
        """경로 데이터를 파일로 저장"""
        route_name = route_data['route_info']['name'].replace(' ', '_').replace('→', 'to')
        route_dir = Path(base_dir) / route_name
        route_dir.mkdir(parents=True, exist_ok=True)
        
        # GPS 데이터 저장
        gps_df = pd.DataFrame({
            'Time (s)': route_data['gps_data']['time'],
            'Latitude (°)': route_data['gps_data']['latitude'],
            'Longitude (°)': route_data['gps_data']['longitude'],
            'Height (m)': route_data['gps_data']['height'],
            'Velocity (m/s)': route_data['gps_data']['velocity'],
            'Direction (°)': route_data['gps_data']['direction'],
            'H.Accuracy (m)': route_data['gps_data']['h_accuracy'],
            'V.Accuracy (°)': route_data['gps_data']['v_accuracy']
        })
        gps_df.to_csv(route_dir / "Location.csv", index=False)
        
        # 가속도계 데이터 저장
        acc_df = pd.DataFrame({
            'Time (s)': route_data['sensor_data']['time'],
            'X (m/s^2)': route_data['sensor_data']['acc_x'],
            'Y (m/s^2)': route_data['sensor_data']['acc_y'],
            'Z (m/s^2)': route_data['sensor_data']['acc_z']
        })
        acc_df.to_csv(route_dir / "Accelerometer.csv", index=False)
        
        # 자이로스코프 데이터 저장
        gyro_df = pd.DataFrame({
            'Time (s)': route_data['sensor_data']['time'],
            'X (rad/s)': route_data['sensor_data']['gyro_x'],
            'Y (rad/s)': route_data['sensor_data']['gyro_y'],
            'Z (rad/s)': route_data['sensor_data']['gyro_z']
        })
        gyro_df.to_csv(route_dir / "Gyroscope.csv", index=False)
        
        print(f"💾 {route_name} 데이터 저장 완료: {route_dir}")
        return route_dir

def main():
    """메인 실행 함수"""
    print("🚀 경로 시뮬레이션 데이터 생성기")
    print("=" * 60)
    
    simulator = RouteSimulator()
    
    # 경로 네트워크 생성
    routes = simulator.generate_route_network()
    
    print(f"📍 {len(routes)}개 경로 시뮬레이션 시작\n")
    
    generated_routes = []
    
    for route_id, route_info in routes.items():
        # 경로 데이터 생성
        route_data = simulator.create_route_data(route_info)
        
        # 데이터 저장
        saved_path = simulator.save_route_data(route_data)
        generated_routes.append({
            'id': route_id,
            'path': saved_path,
            'info': route_info
        })
        
        print(f"   - 거리: {route_info['distance']}m")
        print(f"   - 예상 시간: {route_info['duration']/60:.1f}분")
        print(f"   - 난이도: {route_info['difficulty_type']}\n")
    
    print("✅ 모든 경로 시뮬레이션 완료!")
    print(f"📁 저장 위치: data/simulated_routes/")
    
    # 경로 요약 정보 저장
    summary = []
    for route in generated_routes:
        summary.append({
            'route_id': route['id'],
            'name': route['info']['name'],
            'difficulty_type': route['info']['difficulty_type'],
            'distance_m': route['info']['distance'],
            'duration_s': route['info']['duration'],
            'data_path': str(route['path'])
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("data/simulated_routes/route_summary.csv", index=False)
    print(f"📋 경로 요약: data/simulated_routes/route_summary.csv")

if __name__ == "__main__":
    main()