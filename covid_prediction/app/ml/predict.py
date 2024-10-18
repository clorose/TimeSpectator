import torch
from app.ml.model import CovidPredictor
from app.core.config import MODEL_FILE, SEQ_LENGTH, MIN, MAX

model = CovidPredictor()
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()

def predict_next_day(recent_cases):
    if len(recent_cases) != SEQ_LENGTH:
        raise ValueError(f"Input must contain {SEQ_LENGTH} days of data")

    input_seq = torch.FloatTensor(recent_cases).view(1, SEQ_LENGTH, 1)
    input_seq = (input_seq - MIN) / (MAX - MIN)  # Normalize

    with torch.no_grad():
        model.reset_hidden_state()
        prediction = model(input_seq)
        prediction = prediction.item() * (MAX - MIN) + MIN  # Denormalize

    return round(prediction, 2)
