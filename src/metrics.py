import numpy as np
from math import nan
from pyrender.mesh import Mesh

from .generate import create_model, set_shape, GENDER_TO_STR_DICT
from .load import MeshMeasurements


# Allowable errors.
ALLOW_ERR = np.array([5.0, 11.0, 15.0, 12.0, 12.0, nan, nan, nan, 6.0, nan, 4.0, nan, nan, nan, 8.0, 10.0]) * 0.001


def params_error(pred_params, gt_params):
    return np.abs(pred_params - gt_params)


def measurement_error(pred_meas_obj, gt_meas_obj):
    pred_meas = pred_meas_obj.allmeasurements
    gt_meas = gt_meas_obj.allmeasurements
    return np.abs(pred_meas - gt_meas)


def surface2surface_dist(pred_verts, gt_verts):
    return np.linalg.norm(pred_verts - gt_verts, axis=1)


def evaluate(y_predict, y_target, genders, mode='measurements', poses=None):
    if mode == 'measurements':
        #return np.zeros(10), np.zeros(10), np.zeros(10), np.mean(np.abs(y_predict - y_target), axis=0), \
        #    np.std(np.abs(y_predict - y_target), axis=0), np.max(np.abs(y_predict - y_target), axis=0), np.empty(0), np.empty(0), np.empty(0)
        maes = np.abs(y_predict - y_target)
        mres = maes / y_target
        allowable_ratios = (maes < ALLOW_ERR).sum(axis=0) / maes.shape[0]
        return np.zeros((y_predict.shape[0], 10)), maes, np.empty(0), mres, allowable_ratios
    else:
        params_errors = []
        #measurement_errors = []
        predictions = []
        targets = []
        surface2surface_dists = []

        for sample_idx, pred_params in enumerate(y_predict):
            gender = GENDER_TO_STR_DICT[genders[sample_idx]]
            gt_params = y_target[sample_idx].reshape((1, 10))
            pred_params = pred_params.reshape((1, 10))
            pose = poses[sample_idx]

            gt_meas_obj = MeshMeasurements.__init_from_params__(gender, gt_params, keep_mesh=True, pose=pose)
            gt_size = gt_meas_obj.overall_height
            gt_verts = gt_meas_obj.verts

            pred_meas_obj = MeshMeasurements.__init_from_params__(gender, pred_params, mesh_size=gt_size, keep_mesh=True, pose=pose)
            pred_verts = pred_meas_obj.verts

            params_errors.append(params_error(pred_params[:, 0], gt_params[:, 0]))
            #measurement_errors.append(measurement_error(pred_meas_obj, gt_meas_obj))
            predictions.append(pred_meas_obj.allmeasurements)
            targets.append(gt_meas_obj.allmeasurements)
            surface2surface_dists.append(surface2surface_dist(pred_verts, gt_verts))

            gt_meas_obj.flush()
            pred_meas_obj.flush()

        params_errors = np.array(params_errors)
        predictions = np.array(predictions)
        targets = np.array(targets)
        maes = np.abs(predictions - targets)
        mres = maes / targets
        allowable_ratios = (maes < MeshMeasurements.AEs).sum(axis=0) / maes.shape[0]
        surface2surface_dists = np.array(surface2surface_dists)
        # TODO: MRE, AEs
        #mres = maes / 

        #params_means = np.mean(params_errors, axis=0)
        #params_stds = np.std(params_errors, axis=0)

        #measurement_means = np.mean(measurement_errors, axis=0)
        #measurement_stds = np.std(measurement_errors, axis=0)

        #surface2surface_means = np.mean(surface2surface_dists, axis=0)
        #surface2surface_stds = np.std(surface2surface_dists, axis=0)

        return params_errors, maes, surface2surface_dists, mres, allowable_ratios
