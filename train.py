import argparse
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from prepare import prepare
from metrics import evaluate


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

    @staticmethod
    def feature_importances(model):
        if type(model) == MLPRegressor:
            return None
        elif type(model) == DecisionTreeRegressor:
            return model.feature_importances_
        else:
            return model.coef_


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

    print(f'Preparing {args.dataset_name} dataset...')
    X, y, measurements, genders = prepare(args)
    print('Train/test splitting...')
    X_train, X_test, y_train, y_test, _, gt_meas_test, _, gender_test = train_test_split(
        X, y, measurements, genders, test_size=0.33, random_state=42)

    print(f'Creating {args.model} model...')
    model = getattr(Models, args.model)()
    print(f'Fitting model using {args.regressor_type} regressor...')
    reg = model.fit(X_train, y_train)
    print('Predicting...')
    y_predict = reg.predict(X_test)
    print('Evaluating...')
    params_errors, measurement_errors, s2s_errors = evaluate(y_predict, y_test, gt_meas_test, gender_test)

    print(f'Average absolute error: {params_errors}')
