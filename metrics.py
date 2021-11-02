import numpy as np

from generate import create_model, set_shape, GENDER_TO_STR_DICT
from load import MeshMeasurements


def params_error(pred_params, gt_params):
    return np.abs(pred_params - gt_params)


def measurement_error(pred_meas_obj, gt_meas_obj):
    pred_meas = pred_meas_obj.allmeasurements
    gt_meas = gt_meas_obj.allmeasurements
    return np.abs(pred_meas - gt_meas)


def surface2surface_dist(pred_verts, gt_verts):
    return np.linalg.norm(pred_verts - gt_verts, axis=1)


def evaluate(y_predict, y_target, genders, mode='measurements'):
    if mode == 'measurements':
        #return np.zeros(10), np.zeros(10), np.zeros(10), np.mean(np.abs(y_predict - y_target), axis=0), \
        #    np.std(np.abs(y_predict - y_target), axis=0), np.max(np.abs(y_predict - y_target), axis=0), np.empty(0), np.empty(0), np.empty(0)
        return np.zeros((y_predict.shape[0], 10)), np.abs(y_predict - y_target), np.empty(0)
    else:
        params_errors = []
        measurement_errors = []
        surface2surface_dists = []

        for sample_idx, pred_params in enumerate(y_predict):
            gender = GENDER_TO_STR_DICT[genders[sample_idx]]
            gt_params = y_target[sample_idx].reshape((1, 10))
            pred_params = pred_params.reshape((1, 10))

            gt_meas_obj = MeshMeasurements.__init_from_shape__(gender, gt_params, keep_mesh=True)
            gt_size = gt_meas_obj.overall_height
            gt_verts = gt_meas_obj.verts

            pred_meas_obj = MeshMeasurements.__init_from_shape__(gender, pred_params, mesh_size=gt_size, keep_mesh=True)
            pred_verts = pred_meas_obj.verts

            params_errors.append(params_error(pred_params[:, 0], gt_params[:, 0]))
            measurement_errors.append(measurement_error(pred_meas_obj, gt_meas_obj))
            surface2surface_dists.append(surface2surface_dist(pred_verts, gt_verts))

            gt_meas_obj.flush()
            pred_meas_obj.flush()

        params_errors = np.array(params_errors)
        measurement_errors = np.array(measurement_errors)
        surface2surface_dists = np.array(surface2surface_dists)

        #params_means = np.mean(params_errors, axis=0)
        #params_stds = np.std(params_errors, axis=0)

        #measurement_means = np.mean(measurement_errors, axis=0)
        #measurement_stds = np.std(measurement_errors, axis=0)

        #surface2surface_means = np.mean(surface2surface_dists, axis=0)
        #surface2surface_stds = np.std(surface2surface_dists, axis=0)

        return params_errors, measurement_errors, surface2surface_dists
