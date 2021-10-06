import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

    print('Preparing dataset...')
    X, y, measurements = prepare(args)
    print('Train/test splitting...')
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, measurements, test_size=0.33, random_state=42)

    print('Creating model...')
    model = LinearRegression()
    print('Fitting model...')
    reg = model.fit(X_train, y_train)
    print('Predicting...')
    y_predict = reg.predict(X_test)

    error = np.mean(np.abs(y_predict - y_test))
    print(f'Average absolute error: {error}')
