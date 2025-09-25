import pandas as pd
import numpy as np

def load_sensor_data(data_dir):
    """센서 데이터 로드 유틸리티"""
    files = {
        'accel': f'{data_dir}/Accelerometer.csv',
        'gyro': f'{data_dir}/Gyroscope.csv',
        'gravity': f'{data_dir}/Gravity.csv',
        'linear_accel': f'{data_dir}/Linear Accelerometer.csv'
    }

    data = {}
    for key, file_path in files.items():
        try:
            df = pd.read_csv(file_path)
            print(f"{key}: {len(df)} rows")
            data[key] = df
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
            continue

    return data

def combine_sensor_data(data):
    """가속도계와 자이로스코프 데이터 결합"""
    accel = data['accel'].rename(columns={
        'X (m/s^2)': 'ax', 'Y (m/s^2)': 'ay', 'Z (m/s^2)': 'az'
    })
    gyro = data['gyro'].rename(columns={
        'X (rad/s)': 'gx', 'Y (rad/s)': 'gy', 'Z (rad/s)': 'gz'
    })

    # 시간 기준으로 병합
    combined = pd.merge(accel, gyro, on='Time (s)', how='inner')
    print(f"Combined data: {len(combined)} rows")

    return combined