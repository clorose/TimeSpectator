from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
    recent_cases: List[int]

class PredictionOutput(BaseModel):
    predicted_cases: float
