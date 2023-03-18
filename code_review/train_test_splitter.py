import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

import yaml


def parse_args():
    parser = ArgumentParser('Train test split')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input dataset')
    parser.add_argument('--output_train', required=True, help='Path to train')
    parser.add_argument('--output_test', required=True, help='Path to test')
    parser.add_argument('--params', required=True, help='Path to params file')
    return parser.parse_args()


def main(args):
    with open(args.params, 'r') as fp:
        params = yaml.safe_load(fp)

    data = pd.read_csv(args.input, index_col=0)

    df_train, df_test = train_test_split(data, **params['split_test'])

    df_train.to_csv(args.output_train)
    df_test.to_csv(args.output_test)


if __name__ == '__main__':
    args = parse_args()
    main(args)
