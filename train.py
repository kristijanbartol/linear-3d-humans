import argparse
import numpy as np
from sklearn.linear_model import LinearRegression

from prepare import prepare


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root', type=str, help='root data folder')
    parser.add_argument(
        '--dataset_name', type=str, help='dataset name')
    parser.add_argument(
        '--regressor_type', type=str, help='set of input features'
    )

    return parser


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    X, y = prepare(args)

    model = LinearRegression()
    reg = model.fit(X, y)
