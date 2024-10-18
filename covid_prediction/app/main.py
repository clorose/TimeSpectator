from fastapi import FastAPI
from app.api import endpoints

app = FastAPI(title="COVID-19 Prediction API")

app.include_router(endpoints.router)

@app.get("/")
async def root():
    return {"message": "Welcome to COVID-19 Prediction API"}