import numpy as np

from generate import create_model, set_shape, GENDER_TO_STR_DICT
from prepare import MeshMeasurements


def params_error(pred_params, gt_params):
    return np.abs(pred_params - gt_params)


def measurement_error(pred_verts, gt_meas):
    pred_meas = MeshMeasurements(pred_verts).measurements
    return np.abs(pred_meas - gt_meas)


def surface2surface_dist(pred_verts, gt_verts):
    return np.linalg.norm(pred_verts - gt_verts, axis=1)


def evaluate(y_predict, y_target, gt_measures, genders):
    params_errors = []
    measurement_errors = []
    surface2surface_dists = []

    for sample_idx, pred_params in enumerate(y_predict):
        gender = GENDER_TO_STR_DICT[genders[sample_idx]]
        gt_params = y_target[sample_idx]
        gt_meas = gt_measures[sample_idx]
        
        pred_verts = set_shape(create_model(gender), pred_params).vertices.detach().cpu().numpy().squeeze()
        gt_verts = set_shape(create_model(gender), gt_params).vertices.detach().cpu().numpy().squeeze()

        params_errors.append(params_error(pred_params, gt_params))
        measurement_errors.append(measurement_error(pred_verts, gt_meas))
        surface2surface_dists.append(surface2surface_dist(pred_verts, gt_verts))

    params_errors = np.mean(np.array(params_errors), axis=1)
    measurement_errors = np.mean(np.array(measurement_errors), axis=1)
    surface2surface_dists = np.array(measurement_errors)

    return params_errors, measurement_errors, surface2surface_dists
