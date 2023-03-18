from argparse import ArgumentParser
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
import yaml


def parse_args():
    parser = ArgumentParser('Objective function')
    parser.add_argument('--input_train_validate', type=str, required=True,
                        help='Path to train validate')
    parser.add_argument('--input_val', required=True, help='Path to val')
    parser.add_argument('--test', required=True, help='Path to val')
    parser.add_argument('--params', required=True, help='Path to params file')
    return parser.parse_args()


def extract_data(df):
    X, y = df[['text']], df['label']
    return X, y


def make_pool(X, y):
    return Pool(X, y,
                text_features=text_features,
                feature_names=text_features)


args = parse_args()

with open(args.params, 'r') as fp:
    params = yaml.safe_load(fp)

df_train = pd.read_csv(args.input_train_validate, index_col=0)
df_val = pd.read_csv(args.input_val, index_col=0)

X_train, y_train = extract_data(df_train)
X_val, y_val = extract_data(df_val)

text_features = ['text']

train_pool = make_pool(X_train, y_train)
valid_pool = make_pool(X_val, y_val)


def objective(trial):

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
