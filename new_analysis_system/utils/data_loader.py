#!/usr/bin/env python3
"""
데이터 로더 - test 센서 데이터 로딩
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple

def load_sensor_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    센서 데이터 로딩

    Args:
        data_dir: 센서 데이터가 있는 디렉토리

    Returns:
        센서별 데이터프레임 딕셔너리
    """
    sensor_files = {
        'accelerometer': 'Accelerometer.csv',
        'gyroscope': 'Gyroscope.csv',
        'gravity': 'Gravity.csv',
        'linear_acceleration': 'Linear Acceleration.csv'
    }

    sensor_data = {}

    for sensor_name, filename in sensor_files.items():
        file_path = os.path.join(data_dir, filename)

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"✅ {sensor_name}: {len(df)} rows")
                sensor_data[sensor_name] = df
            except Exception as e:
                print(f"❌ {sensor_name} 로딩 실패: {e}")
        else:
            print(f"⚠️  {sensor_name} 파일 없음: {file_path}")

    return sensor_data

def combine_sensor_data(sensor_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    센서 데이터 결합

    Args:
        sensor_data: 센서별 데이터프레임 딕셔너리

    Returns:
        결합된 데이터프레임
    """
    if not sensor_data:
        raise ValueError("센서 데이터가 없습니다")

    # 기본적으로 accelerometer를 기준으로 함
    if 'accelerometer' not in sensor_data:
        raise ValueError("accelerometer 데이터가 필요합니다")

    base_df = sensor_data['accelerometer'].copy()
    base_df.columns = ['time', 'acc_x', 'acc_y', 'acc_z']

    # 다른 센서 데이터 추가
    if 'gyroscope' in sensor_data:
        gyro_df = sensor_data['gyroscope'].copy()
        gyro_df.columns = ['time', 'gyro_x', 'gyro_y', 'gyro_z']
        base_df = pd.merge(base_df, gyro_df, on='time', how='inner')

    if 'gravity' in sensor_data:
        gravity_df = sensor_data['gravity'].copy()
        gravity_df.columns = ['time', 'grav_x', 'grav_y', 'grav_z']
        base_df = pd.merge(base_df, gravity_df, on='time', how='inner')

    if 'linear_acceleration' in sensor_data:
        linear_df = sensor_data['linear_acceleration'].copy()
        linear_df.columns = ['time', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z']
        base_df = pd.merge(base_df, linear_df, on='time', how='inner')

    print(f"✅ 결합된 데이터: {len(base_df)} rows, {len(base_df.columns)} columns")
    return base_df

def get_data_info(df: pd.DataFrame) -> Dict:
    """
    데이터 기본 정보 추출

    Args:
        df: 센서 데이터프레임

    Returns:
        데이터 정보 딕셔너리
    """
    if df.empty:
        return {}

    time_col = 'time'
    duration = df[time_col].max() - df[time_col].min()
    sampling_rate = len(df) / duration if duration > 0 else 0

    info = {
        'total_samples': len(df),
        'duration_seconds': duration,
        'sampling_rate_hz': sampling_rate,
        'start_time': df[time_col].min(),
        'end_time': df[time_col].max(),
        'columns': list(df.columns)
    }

    return info

if __name__ == "__main__":
    # 테스트 실행
    data_dir = "data/test 2025-09-22 18-30-21"

    print("🔍 센서 데이터 로딩 테스트")
    print("=" * 50)

    sensor_data = load_sensor_data(data_dir)

    if sensor_data:
        combined_df = combine_sensor_data(sensor_data)
        info = get_data_info(combined_df)

        print(f"\n📊 데이터 정보:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        print(f"\n📋 데이터 샘플 (처음 5행):")
        print(combined_df.head())
    else:
        print("❌ 데이터 로딩 실패")