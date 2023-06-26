import pytest
import pandas as pd
from fastapi.testclient import TestClient
from main import app
import requests_mock

@pytest.fixture(scope='session')
def client():
    with requests_mock.Mocker() as rm:
        rm.get('http://localhost', json={})
        with TestClient(app) as client:

            yield client
            

def test_predict_item(client):

    response = client.post("/predict_item", json={'text': 'it was cool'})
    pred = response.json()
    
    assert response.status_code == 200
    assert isinstance(pred, float)


def test_predict_items(client):

    response = client.post("/predict_items", 
                           json=[{'text': 'it was cool'}, {'text': 'it was very awful'}])
    preds = response.json()
    
    assert response.status_code == 200
    assert isinstance(preds, list)


@pytest.fixture(scope="session")
def create_files(tmp_path_factory):

    df_tmp = pd.DataFrame(
        {'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'text': ['like that', 'no worries', 'how much is the fish', 'crazy day',
                 'its raisin now', 'no worries', 'no worries', 'no worries', 
                 'no worries', 'no worries'],
        'label': [1, 1, 0, 1, 2, 1, 1, 1, 1, 1],
        'text_lemm': [[], [], [], [], [], [], [], [], [], []]})
    
    tempdir = tmp_path_factory.mktemp('tmp2', numbered=False)
    (tempdir / 'dopp_train.csv').touch()
    file_name = f'{str(tempdir)}/dopp_train.csv'
    df_tmp.to_csv(file_name, index=False)

    yield tempdir


def test_retrain_model(create_files, client):
    with open('./static/lib/dop_data.csv', 'rb') as f:
        response = client.post("/retrain_model", files={"uploaded_file": ("dop_data.csv", f)})
        assert response.status_code == 200
