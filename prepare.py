import os
from scipy.io import loadmat
from cv2 import imread
import json
from pathlib import Path
import trimesh
import numpy as np

from human_body_prior.tools.omni_tools import colors

from generate import SMPL_NUM_KPTS, create_model, set_shape, GENDER_TO_INT_DICT
from utils import img_to_silhouette


def prepare_caesar():
    data_dir = '/media/kristijan/kristijan-hdd-ex/datasets/NOMO'
    est_kptss_dir = os.path.join(data_dir, 'keypoints', '{}', 'front')
    params_dir = os.path.join(data_dir, 'smple_lbfgsb_params', '{}')
    front_silhss_template = os.path.join(data_dir, 'rendered_rgb', '{}', 'front', 'silh')
    side_silhss_template = os.path.join(data_dir, 'rendered_rgb', '{}', 'side', 'silh')

    save_dir = os.path.join('data', 'caesar', 'prepared')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    data_dict = {
        'genders': [],
        'est_kptss': [],
        'gt_kptss': [],
        'poses': [],
        'shapes': [],
        'front_silhss': [],
        'side_silhss': [],
        'vertss': [],
        'volumes': []
    }

    for gender, num_samples in zip(['male', 'female'], [1474, 2675]):
        for sample_idx in range(num_samples):
            print(gender, sample_idx)
            try:
                data_dict['genders'].append(GENDER_TO_INT_DICT[gender])
                data_dict['poses'].append(loadmat(
                    os.path.join(params_dir.format(gender), f'{sample_idx:04d}.mat'))['pose'])
                data_dict['shapes'].append(loadmat(
                    os.path.join(params_dir.format(gender), f'{sample_idx:04d}.mat'))['shape'])

                smpl_model = create_model(gender, np.zeros([1, SMPL_NUM_KPTS * 3]))
                smpl_output = set_shape(smpl_model, data_dict['shapes'][-1][0])
                verts = smpl_output.vertices.detach().cpu().numpy().squeeze()
                faces = smpl_model.faces.squeeze()
                body_mesh = trimesh.Trimesh(vertices=verts, faces=faces, 
                    vertex_colors=np.tile(colors['grey'], (6890, 1)))

                with open(os.path.join(est_kptss_dir.format(gender), f'{sample_idx:04d}_keypoints.json')) as json_f:
                    data_dict['est_kptss'].append(json.load(json_f)['people'][0]['pose_keypoints_2d'])
                data_dict['gt_kptss'].append(smpl_output.joints.detach().cpu().numpy().squeeze())
                data_dict['front_silhss'].append(img_to_silhouette(imread(
                    os.path.join(front_silhss_template.format(gender), f'{sample_idx:04d}.png'))))
                data_dict['side_silhss'].append(img_to_silhouette(imread(
                    os.path.join(side_silhss_template.format(gender), f'{sample_idx:04d}.png'))))
                data_dict['vertss'].append(verts)
                data_dict['volumes'].append(body_mesh.volume)
            except FileNotFoundError as e:
                print(f'Error with {gender} {sample_idx} ({e.filename})')

    for key in data_dict:
        data_dict[key] = np.array(data_dict[key], dtype=np.float32)
        np.save(os.path.join(save_dir, f'{key}.npy'), data_dict[key])


def prepare_gt(dataset_name):
    ATTR_MAP = {
        'est_kptss': 'joints',
        'genders': 'gender',
        'gt_kptss': 'joints',
        'poses': 'pose',
        'shapes': 'shape',
        'front_silhs': 'front_silhouette',
        'side_silhs': 'side_silhouette',
        'vertss': 'verts',
        'volumes': 'volume'
    }

    data_dir = os.path.join('data', dataset_name, 'generated')
    kpts_dir = os.path.join('data', dataset_name, 'keypoints')  # for est_kptss
    save_dir = os.path.join('data', dataset_name, 'prepared')

    Path(kpts_dir).mkdir(parents=True, exist_ok=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    data_dict = {
        'genders': [],
        'est_kptss': [],
        'gt_kptss': [],
        'poses': [],
        'shapes': [],
        'front_silhs': [],
        'side_silhs': [],
        'vertss': [],
        'volumes': []
    }

    for subj_dirname in os.listdir(data_dir):
        print(subj_dirname)
        subj_dirpath = os.path.join(data_dir, subj_dirname)

        for key in data_dict:
            data = np.load(os.path.join(subj_dirpath, f'{ATTR_MAP[key]}.npy'))
            data_dict[key].append(data)
        # TODO: Use estimated keypoints (OpenPose).

    for key in data_dict:
        data_dict[key] = np.array(data_dict[key], dtype=np.float32)
        np.save(os.path.join(save_dir, f'{key}.npy'), data_dict[key])


if __name__ == '__main__':
    prepare_caesar()
    #prepare_gt('sensors-male')
