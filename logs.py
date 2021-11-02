import numpy as np
import os
from generate import GENDER_TO_STR_DICT

from load import MeshMeasurements, Regressor
from models import Models


RESULTS_DIR = './results/'


def log(model, args, params_errors, measurement_errors, s2s_dists):
    params_means = np.mean(params_errors, axis=0)
    measurement_means = np.mean(measurement_errors, axis=0) * 1000.
    s2s_means = np.mean(s2s_dists, axis=0)

    params_stds = np.std(params_errors, axis=0)
    measurement_stds = np.std(measurement_errors, axis=0) * 1000.
    s2s_stds = np.std(s2s_dists, axis=0)

    params_maxs = np.max(params_errors, axis=0)
    measurement_maxs = np.max(measurement_errors, axis=0) * 1000.
    #s2s_maxs = np.max(s2s_dists, axis=0)

    print('\nPARAMS\n========')
    for param_idx in range(params_means.shape[0]):
        print(f'PCA{param_idx}: {params_means[param_idx]:.6f}, {params_stds[param_idx]:.6f}, {params_maxs[param_idx]:.6f}')

    print('\nMEASURES\n=========')
    measure_labels = MeshMeasurements.alllabels()
    for meas_idx in range(measurement_means.shape[0]):
        print(f'{measure_labels[meas_idx]}: {(measurement_means[meas_idx]):.6f}mm, {(measurement_stds[meas_idx]):.6f}mm, {(measurement_maxs[meas_idx]):.6f}mm')
    print(f'\nMean: {measurement_means.mean():.6f}, {measurement_stds.mean():.6f}')

    # For evaluating others.
    if args is not None:
        print('\nFEATURE IMPORTANCES\n===================')
        feature_labels = Regressor.get_labels(args) + ['w0']
        importances = Models.feature_importances(model)
        for meas_idx in range(len(measure_labels)):
            print(f'{measure_labels[meas_idx]}: {importances[meas_idx]}')


#def log(model, args, params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds):
    #log(model, args, params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds)
    #log_feature_importances(model, args)


def save_results(gender, pred_params, measurement_errors, s2s_dists, subject_idxs):
    gender = GENDER_TO_STR_DICT[gender]
    np.save(os.path.join(RESULTS_DIR, f'{gender}_linear_params.npy'), pred_params)   # NOTE: These are not errors, but estimations.
    np.save(os.path.join(RESULTS_DIR, f'{gender}_linear_measurement_errors.npy'), measurement_errors)
    np.save(os.path.join(RESULTS_DIR, f'{gender}_linear_s2s_errors.npy'), s2s_dists)
    np.save(os.path.join(RESULTS_DIR, f'{gender}_linear_subject_idxs.npy'), subject_idxs)
