import argparse
from copy import deepcopy
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from load import MeshMeasurements, load, load_from_shapes
from metrics import evaluate
from models import Models
from logs import log, save_results
from visualize import visualize, visualize_measurement_distribution


RESULTS_DIR = './results/'
all_to_ap_measurement_idxs = [10, 16, 20, 6, 23, 11, 25, 4, 8, 1, 14, 21, 5, 0, 19]


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
        '--features', type=str, choices=['baseline', 'measurements'],
        help='input features (for fitting from measurements to PCs)'
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
        '--weight_noise2', type=float, help='std added to weight GT'
    )
    parser.add_argument(
        '--num_interaction', type=int, help='# interaction terms added to linear model'
    )

    return parser


if __name__ == '__main__':
    np.random.seed(2021)
    parser = init_argparse()
    args = parser.parse_args()

    print(f'Preparing {args.dataset_name} dataset...')
    #if args.dataset_name != 'star':
    #    X, y, measurements, genders = load_from_shapes(args)
    #else:
    #X, y, measurements, genders = load_from_shapes(args)
    X, y, measurements, genders = load(args)
    all_inputs = np.concatenate([X, measurements], axis=1)
    #X, y, measurements, genders = load_from_shapes(args)
    print('Train/test splitting...')

    np.save(f'{args.gender}_measurements_{args.weight_noise}_{args.weight_noise2}_{args.height_noise}.npy', np.concatenate([X, measurements], axis=1))
    np.save('measurement_names.npy', np.array(['height', 'weight'] + MeshMeasurements.alllabels()))


    indices = np.arange(X.shape[0])


    assert(args.features == 'baseline' or (args.features == 'measurements' and args.target == 'shape'))
    X = deepcopy(X) if args.features == 'baseline' else all_inputs


    X_train, X_test, y_train, y_test, meas_train, meas_test, _, gender_test, train_indices, test_indices = train_test_split(
        X, y, measurements, genders, indices, test_size=0.33, random_state=2021)

    print(f'Creating model...')
    model = LinearRegression()
    #model = Models.tree()


    print(f'Target variable: {args.target}...')
    reg = model.fit(X_train, y_train if args.target == 'shape' else meas_train)
    print('Predicting...')
    y_predict = reg.predict(X_test)
    
    
    importances = Models.feature_importances(model)
    intercepts = Models.intercepts(model)
    
    if args.target == 'measurements':
        importances = importances[all_to_ap_measurement_idxs]
        intercepts = intercepts[all_to_ap_measurement_idxs]
    
    all_coefs = np.concatenate([importances, intercepts.reshape((-1, 1))], axis=1)
    np.save(os.path.join(RESULTS_DIR, f'{args.gender}_meas_coefs.npy'), all_coefs)
    
    
    print('Evaluating...')
    y_gt = y_test if args.target == 'shape' else meas_test
    params_errors, maes, s2s_dists, mres, allowable_ratios = evaluate(y_predict, y_gt, gender_test, args.target)

    score = r2_score(y_gt, y_predict)
    print(f'R2-score: {score}')

    # NOTE: Need to predict shape parameters here.
    print('Saving results...')
    save_results(genders[0], y_predict, maes, s2s_dists, test_indices, args)

    print('Logging to stdout...')
    log(model, args, params_errors, maes, s2s_dists, mres, allowable_ratios)
    
    print('Visualizing...')
    visualize_measurement_distribution(args.gender, X, measurements)
    #visualize(model, args, params_errors, measurement_errors, s2s_dists)
