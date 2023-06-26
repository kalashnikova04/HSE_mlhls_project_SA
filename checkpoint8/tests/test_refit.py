import pandas as pd
from refit import splitter
from objective_catboost import extract_data, make_pool
from catboost import Pool


data = {'id': [1, 2, 3],
        'text': ['like that', 'no worries', 'how much is the fish'],
        'label': [1, 1, 0],
        'text_lemm': [[], [], []]}

df = pd.DataFrame(data)

def test_splitter():

    df.to_csv('./static/lib/testing/test_split.csv')

    train, val = splitter('./static/lib/testing/test_split.csv')

    assert train.shape == (2, 4)
    assert val.shape == (1, 4)


def test_extract():

    X, y = extract_data(df) 

    assert (df[['text']] == X).all()[0]
    assert (df.label == y).all()


def test_pool():

    pool = make_pool(*extract_data(df))
    assert isinstance(pool, Pool)
