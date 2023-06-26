
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import yaml
from objective_catboost import objective, extract_data, make_pool
from sklearn.metrics import f1_score


def splitter(data):
    with open('params.yaml', 'r') as fp:
        params = yaml.safe_load(fp)

    data = pd.read_csv(data, index_col=0)

    df_train, df_val = train_test_split(data, **params['split_val'])

    df_train.to_csv('./static/lib/dop_train.csv')
    df_val.to_csv('./static/lib/dop_val.csv')

    return df_train, df_val


def refit(data):
    with open('params.yaml', 'r') as fp:
        params = yaml.safe_load(fp)
    
    df_train, df_val = splitter(data)

    X_train, y_train = extract_data(df_train)
    X_val, y_val = extract_data(df_val)

    train_pool = make_pool(X_train, y_train)
    valid_pool = make_pool(X_val, y_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15, timeout=6000)

    catboost_params = {
        'iterations': study.best_trial.params['iterations'],
        'depth': 8,
        'min_data_in_leaf': study.best_trial.params['min_data_in_leaf'],
        'learning_rate': study.best_trial.params['learning_rate'],
        'early_stopping_rounds': study.best_trial.params['early_stopping_rounds'],
        **params['catboost']}

    model = CatBoostClassifier(**catboost_params)
    model.fit(train_pool, eval_set=valid_pool)

    return model
