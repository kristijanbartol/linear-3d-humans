import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from prepare import prepare


class Models():

    RANDOM_STATE = 37

    @staticmethod
    def linear():
        return LinearRegression()

    @staticmethod
    def poly():
        return make_pipeline(PolynomialFeatures(degree=5), LinearRegression())

    @staticmethod
    def tree():
        return DecisionTreeRegressor(random_state=Models.RANDOM_STATE)

    @staticmethod
    def mlp():
        return MLPRegressor(hidden_layer_sizes=(2000), random_state=Models.RANDOM_STATE, max_iter=500)


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root', type=str, 
        help='root data folder')
    parser.add_argument(
        '--dataset_name', type=str, 
        help='dataset name')
    parser.add_argument(
        '--regressor_type', type=str,
        help='regressor type defined by the set of input features'
    )
    parser.add_argument(
        '--model', type=str, choices=['linear', 'poly', 'tree', 'mlp'],
        help='machine learning model type'
    )

    return parser


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    print('Preparing dataset...')
    X, y, measurements = prepare(args)
    print('Train/test splitting...')
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print('Creating model...')
    model = getattr(Models, args.model)()
    print('Fitting model...')
    reg = model.fit(X_train, y_train)
    print('Predicting...')
    y_predict = reg.predict(X_test)

    error = np.mean(np.abs(y_predict - y_test))
    print(f'Average absolute error: {error}')
