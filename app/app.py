from fastapi import FastAPI, Request 
from pydantic import BaseModel
from model import predict_batch

app = FastAPI()

class InputText(BaseModel): 
    texts = list[str]

@app.post("/predict")
def predict_sentiment(input_data: InputText): 
    results = predict_batch(input_data.texts) 
    return results 