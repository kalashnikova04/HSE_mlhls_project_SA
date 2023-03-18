import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import yaml


def parse_args():
    parser = ArgumentParser('Train val split')
    parser.add_argument('--train', type=str, required=True,
                        help='Path to input dataset')
    parser.add_argument('--output_train_validate', required=True,
                        help='Path to train for validation')
    parser.add_argument('--output_val', required=True, help='Path to val')
    parser.add_argument('--params', required=True, help='Path to params file')
    return parser.parse_args()


def main(args):
    with open(args.params, 'r') as fp:
        params = yaml.safe_load(fp)

    data = pd.read_csv(args.train, index_col=0)

    df_train, df_test = train_test_split(data, **params['split_val'])

    df_train.to_csv(args.output_train_validate)
    df_test.to_csv(args.output_val)


if __name__ == '__main__':
    args = parse_args()
    main(args)
