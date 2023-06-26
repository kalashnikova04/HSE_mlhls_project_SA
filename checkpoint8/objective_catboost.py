import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
import yaml


def extract_data(df):
    X, y = df[['text']], df['label']
    return X, y


def make_pool(X, y):
    return Pool(X, y,
                text_features=text_features,
                feature_names=text_features)


def objective(trial):

    with open('params.yaml', 'r') as fp:
        params = yaml.safe_load(fp)

    df_train = pd.read_csv('./static/lib/dop_train.csv', index_col=0)
    df_val = pd.read_csv('./static/lib/dop_val.csv', index_col=0)

    X_train, y_train = extract_data(df_train)
    X_val, y_val = extract_data(df_val)

    train_pool = make_pool(X_train, y_train)
    valid_pool = make_pool(X_val, y_val)

    catboost_params = {
        "iterations": trial.suggest_int("iterations", 1000, 3000),
        "depth": trial.suggest_int("depth", 2, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        'task_type': 'GPU',
        'early_stopping_rounds': trial.suggest_int("early_stopping_rounds",
                                                   100, 1000),
        **params['catboost']
    }

    text_processing = params['text_processing']

    model = CatBoostClassifier(**catboost_params,
                               text_processing=text_processing)
    model.fit(train_pool, eval_set=valid_pool)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average='macro')

    return f1

text_features = ['text']
