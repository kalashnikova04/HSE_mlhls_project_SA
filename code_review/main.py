from argparse import ArgumentParser
import json
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from mlflow.catboost import log_model
import yaml
from objective_catboost import objective, extract_data, make_pool
from sklearn.metrics import f1_score


def parse_args():
    parser = ArgumentParser('Train')
    parser.add_argument('--input_train_validate', type=str, required=True,
                        help='Path to train validate')
    parser.add_argument('--input_val', required=True, help='Path to val')
    parser.add_argument('--test', required=True, help='Path to test')
    parser.add_argument('--params', required=True, help='Path to params file')
    return parser.parse_args()


def main(args):
    with open(args.params, 'r') as fp:
        params = yaml.safe_load(fp)

    df_train = pd.read_csv(args.input_train_validate, index_col=0)
    df_val = pd.read_csv(args.input_val, index_col=0)
    df_test = pd.read_csv(args.test, index_col=0)

    X_train, y_train = extract_data(df_train)
    X_val, y_val = extract_data(df_val)
    X_test, y_test = extract_data(df_test)

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

    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
    test_f1_micro = f1_score(y_test, y_test_pred, average='micro')

    train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
    train_f1_micro = f1_score(y_train, y_train_pred, average='micro')

    with open('metrics.json', 'w') as fp:
        json.dump({
            'train_f1_macro': train_f1_macro,
            'train_f1_micro': train_f1_micro,
            'test_f1_macro': test_f1_macro,
            'test_f1_micro': test_f1_micro
        }, fp)
    
    log_model(cb_model)


if __name__ == '__main__':
    args = parse_args()
    main(args)
