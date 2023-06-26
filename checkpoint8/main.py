import logging
import pickle
import catboost
import pandas as pd
import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from refit import refit
from elastic_logging import es, index_name
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

if os.path.exists('retrained_boost.pkl'):
    model = pickle.load(open('retrained_boost.pkl', 'rb'))
else:
    model = pickle.load(open('model.pickle', 'rb'))

print(model)


class Item(BaseModel):
    text: str

    def predict(self):
        y = model.predict_proba(pd.DataFrame({'text':self.text}, index=[0]))
        return y

class Items(BaseModel):
    objects: List[Item]

@app.get('/')
async def root():
    return {'message': 'start page'}

@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    # Log the request
    logging.info(f"Received request: {item}")

    pred = item.predict()
    # Log the prediction
    logging.info(f"Prediction: {pred.argmax()}")

    es.index(index=index_name, document={
        'timestamp': datetime.utcnow(),
        'event' : 'PREDICTION',
        'text': item.text,
        'proba': pred,
        'class':pred.argmax(),
    })

    return pred.argmax()


@app.post("/predict_items")
async def predict_items(items: List[Item]) -> List[float]:
    results = []

    # Log the requests
    for item in items:
        logging.info(f"Received request: {item}")
        pred = item.predict()
        results.append(pred.argmax())
        es.index(index=index_name, document={
            'timestamp': datetime.utcnow(),
            'event' : 'PREDICTION',
            'text': item.text,
            'proba': pred,
            'class':pred[0],
        })

    # Log the predictions
    logging.info(f"Predictions: {results}")

    return results


@app.post("/retrain_model")
async def retrain_model(uploaded_file: UploadFile = File(...)) -> str:

    filename = uploaded_file.filename

    with open(f'./static/lib/{filename}', mode='wb+') as f:

        shutil.copyfileobj(uploaded_file.file, f)
        uploaded_file.file.close()

    model = refit(f'./static/lib/{filename}')

    with open('retrained_boost.pkl', 'wb') as f:
        pickle.dump(model, f)

    es.index(index=index_name, document={
        'timestamp': datetime.utcnow(),
        'event' : 'RETRAIN',
    })
    return 'Retraining finished'
