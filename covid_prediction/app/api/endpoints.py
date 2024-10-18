from fastapi import APIRouter, HTTPException
from app.ml.predict import predict_next_day
from app.schemas.prediction import PredictionInput, PredictionOutput

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        prediction = predict_next_day(input_data.recent_cases)
        return PredictionOutput(predicted_cases=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
