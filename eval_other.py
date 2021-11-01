import os
import numpy as np
from scipy.io import loadmat

from load import MeshMeasurements
from metrics import evaluate
from logs import log


GT_ROOT = '/media/kristijan/kristijan-hdd-ex/datasets/NOMO/smple_lbfgsb_params/'
GT_PATH_TEMPLATE = os.path.join(GT_ROOT, '{}', '{:04d}.mat')


EXPOSE_ROOT = '/media/kristijan/kristijan-hdd-ex/expose/'
EXPOSE_EST_ROOT = os.path.join(EXPOSE_ROOT, 'output')


def eval_smplify():
    SMPLIFY_PARAMS = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).reshape(1, 10)

    gt_params_all = []

    est_meas_all = []
    gt_meas_all = []

    for gender, num_samples in zip(['male', 'female'], [1474, 2675]):
        for subject_idx in range(num_samples)[:100]:
            print(f'{gender} {subject_idx}/{num_samples}')
            try:
                gtpath = GT_PATH_TEMPLATE.format(gender, subject_idx)
                gt_params = loadmat(gtpath)['shape'].reshape(1, 10)
                gt_params_all.append(gt_params)

                gt_meas_obj = MeshMeasurements.__init_from_shape__(gender, gt_params)
                gt_size = gt_meas_obj.overall_height

                est_meas_obj = MeshMeasurements.__init_from_shape__(gender, SMPLIFY_PARAMS, gt_size)

                gt_meas_all.append(gt_meas_obj.allmeasurements)
                est_meas_all.append(est_meas_obj.allmeasurements)
            except FileNotFoundError as e:
                print(f'Error with {gender} {subject_idx} ({e.filename})')

    est_meas_all = np.array(est_meas_all)
    gt_meas_all = np.array(gt_meas_all)

    params_errors, measurement_errors, s2s_dists = evaluate(est_meas_all, gt_meas_all)

    log(None, None, params_errors, measurement_errors, s2s_dists)


def eval_expose():
    est_params_all = []
    gt_params_all = []

    est_meas_all = []
    gt_meas_all = []

    for dirname in os.listdir(EXPOSE_EST_ROOT)[:100]:
        print(dirname)
        fname = f'{dirname}_params.npz'
        fpath = os.path.join(EXPOSE_EST_ROOT, dirname, fname)

        gender = dirname.split('_')[0]
        subject_idx = dirname.split('_')[1].split('.')[0]

        out_data = np.load(fpath)
        est_params = out_data['betas'].reshape(1, 10)
        est_params_all.append(est_params)

        gtpath = GT_PATH_TEMPLATE.format(gender, str(subject_idx))

        gt_params = loadmat(gtpath)['shape'].reshape(1, 10)
        gt_params_all.append(gt_params)

        gt_meas_obj = MeshMeasurements.__init_from_shape__(gender, gt_params)
        gt_size = gt_meas_obj.overall_height

        est_meas_obj = MeshMeasurements.__init_from_shape__(gender, est_params, gt_size)

        gt_meas_all.append(gt_meas_obj.allmeasurements)
        est_meas_all.append(est_meas_obj.allmeasurements)

    est_meas_all = np.array(est_meas_all)
    gt_meas_all = np.array(gt_meas_all)

    params_errors, measurement_errors, s2s_dists = evaluate(est_meas_all, gt_meas_all)

    print(measurement_errors.mean())



if __name__ == '__main__':
    eval_smplify()
    #eval_expose()
