# File: C:\gaon\2024\covid_prediction\app\ml\data_preprocessing.py
# 목적: 이 파일은 COVID-19 데이터의 전처리를 담당합니다.
# 데이터 로딩, 시퀀스 생성, 데이터 분할 및 스케일링 기능을 제공합니다.

import pandas as pd
import numpy as np
import torch
from app.core.config import DATA_FILE, SEQ_LENGTH

def load_and_preprocess_data():
    """
    COVID-19 확진자 데이터를 로드하고 전처리합니다.
    
    반환값:
        pd.Series: 대한민국의 일일 확진자 수
    """
    # CSV 파일에서 확진자 데이터 로드
    confirmed = pd.read_csv(DATA_FILE)
    # 대한민국 데이터만 추출
    korea = confirmed[confirmed['Country/Region']=='Korea, South'].iloc[:,4:].T
    korea.index = pd.to_datetime(korea.index)
    # 일일 확진자 수 계산
    daily_cases = korea.diff().fillna(korea.iloc[0]).astype('int')
    return daily_cases

def create_sequences(data, seq_length):
    """
    시계열 예측을 위한 입력 시퀀스와 타겟 값을 생성합니다.
    
    인자:
        data (pd.Series): 시계열 데이터
        seq_length (int): 입력 시퀀스의 길이
    
    반환값:
        tuple: (입력 시퀀스 배열, 타겟 값 배열)
    """
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i+seq_length)]
        y = data.iloc[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def min_max_scale(array, min_val, max_val):
    """
    주어진 배열에 대해 Min-Max 스케일링을 수행합니다.
    
    인자:
        array (np.array): 스케일링할 배열
        min_val (float): 스케일링 전 최소값
        max_val (float): 스케일링 전 최대값
    
    반환값:
        np.array: 스케일링된 배열
    """
    return (array - min_val) / (max_val - min_val)

def make_tensor(array):
    """
    NumPy 배열을 PyTorch 텐서로 변환합니다.
    
    인자:
        array (np.array): 변환할 NumPy 배열
    
    반환값:
        torch.Tensor: 변환된 PyTorch 텐서
    """
    return torch.from_numpy(array).float()

def split_and_scale_data(X, y):
    """
    데이터를 훈련, 검증, 테스트 세트로 분할하고 스케일링합니다.
    
    인자:
        X (np.array): 입력 시퀀스
        y (np.array): 타겟 값
    
    반환값:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, MIN, MAX)
    """
    # 데이터를 훈련, 검증, 테스트 세트로 분할
    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # 스케일링을 위한 최소값과 최대값 계산
    MIN = X_train.min()
    MAX = X_train.max()

    # 데이터 스케일링
    X_train = min_max_scale(X_train, MIN, MAX)
    y_train = min_max_scale(y_train, MIN, MAX)
    X_val = min_max_scale(X_val, MIN, MAX)
    y_val = min_max_scale(y_val, MIN, MAX)
    X_test = min_max_scale(X_test, MIN, MAX)
    y_test = min_max_scale(y_test, MIN, MAX)

    # NumPy 배열을 PyTorch 텐서로 변환
    X_train = make_tensor(X_train)
    y_train = make_tensor(y_train)
    X_val = make_tensor(X_val)
    y_val = make_tensor(y_val)
    X_test = make_tensor(X_test)
    y_test = make_tensor(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, MIN, MAX