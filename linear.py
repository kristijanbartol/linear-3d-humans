import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from load import load, load_from_shapes
from metrics import evaluate
from models import Models
from logs import log, save_results
from visualize import visualize


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root', type=str, 
        help='root data folder'
    )
    parser.add_argument(
        '--dataset_name', type=str, 
        help='dataset name'
    )
    parser.add_argument(
        '--target', type=str, choices=['shape', 'measurements'],
        help='target variable'
    )
    parser.add_argument(
        '--gender', type=str, choices=['male', 'female', 'neutral', 'both'],
        help='If both, then evaluate gender-specific model on all data. If neutral, use neutral model'
    )
    parser.add_argument(
        '--height_noise', type=float, help='std added to height GT'
    )
    parser.add_argument(
        '--weight_noise', type=float, help='std added to weight GT'
    )
    parser.add_argument(
        '--num_interaction', type=int, help='# interaction terms added to linear model'
    )

    return parser


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    print(f'Preparing {args.dataset_name} dataset...')
    if args.dataset_name != 'star':
        X, y, measurements, genders = load_from_shapes(args)
    else:
        X, y, measurements, genders = load(args)
    print('Train/test splitting...')
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, meas_train, meas_test, _, gender_test, train_indices, test_indices = train_test_split(
        X, y, measurements, genders, indices, test_size=0.33, random_state=2021)

    print(f'Creating model...')
    model = LinearRegression()
    #model = Models.poly()
    print(f'Target variable: {args.target}...')
    reg = model.fit(X_train, y_train if args.target == 'shape' else meas_train)
    print('Predicting...')
    y_predict = reg.predict(X_test)
    print('Evaluating...')
    y_gt = y_test if args.target == 'shape' else meas_test
    params_errors, measurement_errors, s2s_dists = evaluate(y_predict, y_gt, gender_test, args.target)


    score = r2_score(y_gt, y_predict)
    print(f'R2-score: {score}')


    # NOTE: Need to predict shape parameters here.
    print('Saving results...')
    save_results(genders[0], y_predict, measurement_errors, s2s_dists, test_indices, args)

    print('Logging to stdout...')
    log(model, args, params_errors, measurement_errors, s2s_dists)
    #print('Visualizing...')
    #visualize(model, args, params_errors, measurement_errors, s2s_dists)
