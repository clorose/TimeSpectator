# File: C:\gaon\2024\covid_prediction\app\ml\model.py
# 목적: 이 파일은 LSTM 기반의 COVID-19 예측 모델을 정의합니다.

import torch
from torch import nn
from app.core.config import N_FEATURES, N_HIDDEN, SEQ_LENGTH, N_LAYERS

class CovidPredictor(nn.Module):
    def __init__(self):
        """
        LSTM 기반의 COVID-19 예측 모델 초기화
        """
        super(CovidPredictor, self).__init__()
        self.n_hidden = N_HIDDEN
        self.seq_len = SEQ_LENGTH
        self.n_layers = N_LAYERS
        
        # LSTM 층 정의
        self.lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=N_HIDDEN,
            num_layers=N_LAYERS
        )
        
        # 선형 층 정의
        self.linear = nn.Linear(in_features=N_HIDDEN, out_features=1)
    
    def reset_hidden_state(self):
        """
        LSTM의 히든 스테이트를 초기화합니다.
        """
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )
    
    def forward(self, sequences):
        """
        순전파 수행
        
        Args:
            sequences (torch.Tensor): 입력 시퀀스

        Returns:
            torch.Tensor: 예측값
        """
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred