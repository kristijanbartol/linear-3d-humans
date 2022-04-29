import numpy as np
import os
from generate import GENDER_TO_STR_DICT

from load import MeshMeasurements, Regressor
from models import Models


RESULTS_DIR = './results/'
all_to_ap_measurement_idxs = [10, 16, 20, 6, 23, 11, 25, 4, 8, 1, 14, 21, 5, 0, 19]


def log(model, args, params_errors, maes, s2s_dists, mres, allowable_ratios):
    params_means = np.mean(params_errors, axis=0)
    measurement_means = np.mean(maes, axis=0) * 1000.
    s2s_means = np.mean(s2s_dists, axis=0) * 1000.

    params_stds = np.std(params_errors, axis=0)
    measurement_stds = np.std(maes, axis=0) * 1000.
    s2s_stds = np.std(s2s_dists, axis=0) * 1000.

    params_maxs = np.max(params_errors, axis=0)
    measurement_maxs = np.max(maes, axis=0) * 1000.
    #s2s_maxs = np.max(s2s_dists, axis=0)
    
    mres_means = mres.mean(axis=0) * 100.
    allowable_ratios *= 100.

    print('\nPARAMS\n========')
    for param_idx in range(params_means.shape[0]):
        print(f'PCA{param_idx}: {params_means[param_idx]:.6f}, {params_stds[param_idx]:.6f}, {params_maxs[param_idx]:.6f}')

    print('\nMEASURES\n=========')
    measure_labels = MeshMeasurements.alllabels()
    
    #all_to_ap_measurement_idxs = list(range(len(MeshMeasurements.alllabels())))
    ap_labels = np.array(measure_labels)[all_to_ap_measurement_idxs]
    ap_measurement_means = measurement_means[all_to_ap_measurement_idxs]
    ap_measurement_stds = measurement_stds[all_to_ap_measurement_idxs]
    ap_measurement_maxs = measurement_maxs[all_to_ap_measurement_idxs]
    ap_mres_means = mres_means[all_to_ap_measurement_idxs]
    ap_allowable_ratios = allowable_ratios[all_to_ap_measurement_idxs]
    
    for meas_idx in range(ap_measurement_means.shape[0]):
        print(f'{(ap_labels[meas_idx])}: {(ap_measurement_means[meas_idx]):.6f}mm, '
              f'{(ap_measurement_stds[meas_idx]):.6f}mm, {(ap_measurement_maxs[meas_idx]):.6f}mm, '
              f'{(ap_mres_means[meas_idx]):.6f}%, {(ap_allowable_ratios[meas_idx]):.6f}%')
    print(f'\nMean: {ap_measurement_means.mean():.6f}, Std: {ap_measurement_stds.mean():.6f}')
    print(f'Mean S2S: {np.mean(s2s_means)}, Std S2S: {np.mean(s2s_stds)}')

    # For evaluating others.
    if args is not None:
        print('\nFEATURE IMPORTANCES\n===================')
        importances = Models.feature_importances(model)
        if importances.shape[0] == 10:
            intercepts = Models.intercepts(model)
            all_coefs = np.concatenate([importances, intercepts.reshape((-1, 1))], axis=1)
            np.save(os.path.join(RESULTS_DIR, f'{args.gender}_shape_coefs.npy'), all_coefs)
        importances = importances[all_to_ap_measurement_idxs]
        intercepts = Models.intercepts(model)[all_to_ap_measurement_idxs]
        for meas_idx in range(len(ap_labels)):
            print(f'{ap_labels[meas_idx]}: {importances[meas_idx]} {intercepts[meas_idx]}')
        
        all_coefs = np.concatenate([importances, intercepts.reshape((-1, 1))], axis=1)

        np.save(os.path.join(RESULTS_DIR, f'{args.gender}_meas_coefs.npy'), all_coefs)


#def log(model, args, params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds):
    #log(model, args, params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds)
    #log_feature_importances(model, args)


def save_results(gender, pred_params, measurement_errors, s2s_dists, subject_idxs, args=None):
    gender = GENDER_TO_STR_DICT[gender]
    suffix = f'_{args.height_noise}_{args.weight_noise}' if args is not None else ''
    suffix += f'_{args.num_interaction}' if args.num_interaction > 0 else ''

    np.save(os.path.join(RESULTS_DIR, f'{gender}_linear_params{suffix}.npy'), pred_params)   # NOTE: These are not errors, but estimations.
    np.save(os.path.join(RESULTS_DIR, f'{gender}_linear_measurement_errors{suffix}.npy'), measurement_errors)
    np.save(os.path.join(RESULTS_DIR, f'{gender}_linear_s2s_errors{suffix}.npy'), s2s_dists)
    np.save(os.path.join(RESULTS_DIR, f'{gender}_linear_subject_idxs{suffix}.npy'), subject_idxs)
