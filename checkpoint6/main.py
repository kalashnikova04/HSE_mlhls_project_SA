import logging
import pickle
import catboost
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

model = pickle.load(open('model.pickle', 'rb'))

class Item(BaseModel):
    text: str

    def predict(self):
        y = model.predict(pd.DataFrame({'text':self.text}, index=[0]))
        return y

class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # Log the request
    logging.info(f"Received request: {item}")

    pred = item.predict()

    # Log the prediction
    logging.info(f"Prediction: {pred[0]}")

    return pred[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    results = []

    # Log the requests
    for item in items:
        logging.info(f"Received request: {item}")
        results.append(item.predict()[0])

    # Log the predictions
    logging.info(f"Predictions: {results}")

    return results
