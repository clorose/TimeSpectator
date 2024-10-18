# File: C:\gaon\2024\covid_prediction\app\ml\train.py
# 목적: 이 파일은 LSTM 모델의 훈련 과정을 정의합니다.

import torch
from app.core.config import NUM_EPOCHS, VERBOSE, PATIENCE, MODEL_FILE, SEQ_LENGTH
from app.ml.data_preprocessing import load_and_preprocess_data, create_sequences, split_and_scale_data
from app.ml.model import CovidPredictor

def train_model(model, train_data, train_labels, val_data=None, val_labels=None):
    """
    모델을 훈련시키는 함수

    Args:
        model (CovidPredictor): 훈련할 모델
        train_data (torch.Tensor): 훈련 데이터
        train_labels (torch.Tensor): 훈련 레이블
        val_data (torch.Tensor, optional): 검증 데이터
        val_labels (torch.Tensor, optional): 검증 레이블

    Returns:
        tuple: (훈련된 모델, 훈련 손실 기록, 검증 손실 기록)
    """
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_hist = []
    val_hist = []
    
    for t in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for idx, seq in enumerate(train_data): 
            model.reset_hidden_state()
            seq = torch.unsqueeze(seq, 0)
            y_pred = model(seq)
            loss = loss_fn(y_pred[0].float(), train_labels[idx])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_hist.append(epoch_loss / len(train_data))
        
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for val_idx, val_seq in enumerate(val_data):
                    model.reset_hidden_state()
                    val_seq = torch.unsqueeze(val_seq, 0)
                    y_val_pred = model(val_seq)
                    val_step_loss = loss_fn(y_val_pred[0].float(), val_labels[val_idx])
                    val_loss += val_step_loss
                
            val_hist.append(val_loss / len(val_data))
            
            if t % VERBOSE == 0:
                print(f'Epoch {t} train loss: {epoch_loss / len(train_data):.4f} val loss: {val_loss / len(val_data):.4f}')
            
            if (t % PATIENCE == 0) & (t != 0):
                if val_hist[t - PATIENCE] < val_hist[t]:
                    print('\nEarly Stopping')
                    break
        elif t % VERBOSE == 0:
            print(f'Epoch {t} train loss: {epoch_loss / len(train_data):.4f}')
    
    # 모델 저장
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
            
    return model, train_hist, val_hist

def main():
    """
    메인 훈련 함수
    """
    daily_cases = load_and_preprocess_data()
    X, y = create_sequences(daily_cases, SEQ_LENGTH)
    X_train, y_train, X_val, y_val, X_test, y_test, MIN, MAX = split_and_scale_data(X, y)

    model = CovidPredictor()
    trained_model, train_hist, val_hist = train_model(model, X_train, y_train, X_val, y_val)

    print("Training completed.")

if __name__ == "__main__":
    main()