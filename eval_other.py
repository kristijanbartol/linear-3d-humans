import os
import numpy as np
from scipy.io import loadmat
from generate import GENDER_TO_INT_DICT, GENDER_TO_STR_DICT

from load import MeshMeasurements
from metrics import evaluate
from logs import log
from visualize import visualize


GT_ROOT = '/media/kristijan/kristijan-hdd-ex/datasets/NOMO/smple_lbfgsb_params/'
GT_PATH_TEMPLATE = os.path.join(GT_ROOT, '{}', '{:04d}.mat')

RESULTS_DIR = './results'

EXPOSE_ROOT = '/media/kristijan/kristijan-hdd-ex/expose/'
EXPOSE_EST_ROOT = os.path.join(EXPOSE_ROOT, 'output')


def eval_smplify():
    SMPLIFY_PARAMS = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).reshape(1, 10)

    est_params_all = []
    gt_params_all = []

    subject_idxs = []
    genders_all = []

    for gender, num_samples in zip(['male', 'female'], [1474, 2675]):
    #for gender, num_samples in zip(['male'], [1474]):
        for subject_idx in range(num_samples):
            try:
                gtpath = GT_PATH_TEMPLATE.format(gender, subject_idx)
                gt_params = loadmat(gtpath)['shape'].reshape(1, 10)
                gt_params_all.append(gt_params)

                est_params_all.append(SMPLIFY_PARAMS)
                genders_all.append(GENDER_TO_INT_DICT[gender])
                subject_idxs.append(subject_idx)
            except FileNotFoundError as e:
                pass

    gt_params_all = np.array(gt_params_all)
    est_params_all = np.array(est_params_all)
    genders_all = np.array(genders_all, dtype=np.int8)
    subject_idxs = np.array(subject_idxs, dtype=np.int32)

    params_errors, measurement_errors, s2s_dists = evaluate(est_params_all, gt_params_all, genders_all, mode='shapes')

    log(None, None, params_errors, measurement_errors, s2s_dists)
    #visualize(params_errors, measurement_errors, s2s_dists)

    for gender_idx in [0, 1]:
        gender_est_params_all = est_params_all[genders_all == gender_idx]
        gender_measurement_errors = measurement_errors[genders_all == gender_idx]
        gender_s2s_dists = s2s_dists[genders_all == gender_idx]
        gender_subject_idxs = subject_idxs[genders_all == gender_idx]

        gender_str = GENDER_TO_STR_DICT[gender_idx]
        np.save(os.path.join(RESULTS_DIR, f'{gender_str}_smplify_params.npy'), gender_est_params_all)   # NOTE: These are not errors, but estimations.
        np.save(os.path.join(RESULTS_DIR, f'{gender_str}_smplify_measurement_errors.npy'), gender_measurement_errors)
        np.save(os.path.join(RESULTS_DIR, f'{gender_str}_smplify_s2s_errors.npy'), gender_s2s_dists)
        np.save(os.path.join(RESULTS_DIR, f'{gender_str}_smplify_subject_idxs.npy'), gender_subject_idxs)


def eval_expose():
    est_params_all = []
    gt_params_all = []

    genders_all = []
    subject_idxs = []

    for dirname in os.listdir(EXPOSE_EST_ROOT):
        #print(dirname)
        fname = f'{dirname}_params.npz'
        fpath = os.path.join(EXPOSE_EST_ROOT, dirname, fname)

        gender = dirname.split('_')[0]
        subject_idx = int(dirname.split('_')[1].split('.')[0])

        subject_idxs.append(subject_idx)

        out_data = np.load(fpath)
        est_params = out_data['betas'].reshape(1, 10)
        est_params_all.append(est_params)

        gtpath = GT_PATH_TEMPLATE.format(gender, subject_idx)

        gt_params = loadmat(gtpath)['shape'].reshape(1, 10)
        gt_params_all.append(gt_params)

        genders_all.append(GENDER_TO_INT_DICT[gender])

    gt_params_all = np.array(gt_params_all)
    est_params_all = np.array(est_params_all)
    genders_all = np.array(genders_all)
    subject_idxs = np.array(subject_idxs)

    params_errors, measurement_errors, s2s_dists = evaluate(est_params_all, gt_params_all, genders_all, mode='shapes')

    log(None, None, params_errors, measurement_errors, s2s_dists)
    #visualize(params_errors, measurement_errors, s2s_dists)

    for gender_idx in [0, 1]:
        gender_est_params_all = est_params_all[genders_all == gender_idx]
        gender_measurement_errors = measurement_errors[genders_all == gender_idx]
        gender_s2s_dists = s2s_dists[genders_all == gender_idx]
        gender_subject_idxs = subject_idxs[genders_all == gender_idx]

        gender_str = GENDER_TO_STR_DICT[gender_idx]
        np.save(os.path.join(RESULTS_DIR, f'{gender_str}_expose_params.npy'), gender_est_params_all)   # NOTE: These are not errors, but estimations.
        np.save(os.path.join(RESULTS_DIR, f'{gender_str}_expose_measurement_errors.npy'), gender_measurement_errors)
        np.save(os.path.join(RESULTS_DIR, f'{gender_str}_expose_s2s_errors.npy'), gender_s2s_dists)
        np.save(os.path.join(RESULTS_DIR, f'{gender_str}_expose_subject_idxs.npy'), gender_subject_idxs)


if __name__ == '__main__':
    eval_smplify()
    #eval_expose()
