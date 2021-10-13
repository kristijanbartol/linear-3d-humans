from load import MeshMeasurements, Regressor
from models import Models


def log_errors(params, measures, s2s):
    print(f'Mean total errors: [PARAMS: {params.mean():.6f} | MEASURES: {measures.mean():.6f} | S2S: {s2s.mean():.6f}]')

    print('\nPARAMS\n========')
    for param_idx in range(params.shape[0]):
        print(f'PCA{param_idx}: {params[param_idx]:.6f}')

    print('\nMEASURES\n=========')
    measure_labels = MeshMeasurements.labels()
    for meas_idx in range(measures.shape[0]):
        print(f'{measure_labels[meas_idx]}: {measures[meas_idx]:.6f}')


def log_feature_importances(model, args):
    print('\nFEATURE IMPORTANCES\n===================')
    feature_labels = Regressor.get_labels(args)
    importances = Models.feature_importances(model)
    for feat_idx in range(importances.shape[0]):
        print(f'{feature_labels[feat_idx]}: {importances[feat_idx]:.6f}')


def log(model, args, params_errors, measurement_errors, s2s_dists):
    log_errors(params_errors, measurement_errors, s2s_dists)
    log_feature_importances(model, args)
