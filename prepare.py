import os
from scipy.io import loadmat
from cv2 import imread
import json
from pathlib import Path
import trimesh
import pickle
import numpy as np

from human_body_prior.tools.omni_tools import colors

from generate import GENDER_TO_STR_DICT, SMPL_NUM_KPTS, create_model, set_shape, GENDER_TO_INT_DICT
from utils import img_to_silhouette


def process_openpose(pose_json):
    # TODO: Use confidences (could have ones for GT).
    return np.array(pose_json['people'][0]['pose_keypoints_2d']).reshape([-1, 3])[:, :2]


def prepare_caesar():
    data_dir = '/media/kristijan/kristijan-hdd-ex/datasets/NOMO'
    est_kptss_dir = os.path.join(data_dir, 'keypoints', '{}', 'front')
    params_dir = os.path.join(data_dir, 'smple_lbfgsb_params', '{}')
    front_silhss_template = os.path.join(data_dir, 'rendered_rgb', '{}', 'front', 'silh')
    side_silhss_template = os.path.join(data_dir, 'rendered_rgb', '{}', 'side', 'silh')

    save_dir_template = os.path.join('data', 'caesar', 'prepared', '{}')

    data_dict = {
        'genders': [],
        'est_kptss': [],
        'gt_kptss': [],
        'poses': [],
        'shapes': [],
        'front_silhs': [],
        'side_silhs': [],
        'vertss': [],
        'facess': [],
        'volumes': []
    }

    for gender, num_samples in zip(['male', 'female'], [1474, 2675]):
    #for gender, num_samples in zip(['male'], [1474]):
        for sample_idx in range(num_samples):
            print(gender, sample_idx)
            try:
                data_dict['poses'].append(loadmat(
                    os.path.join(params_dir.format(gender), f'{sample_idx:04d}.mat'))['pose'])
                data_dict['shapes'].append(loadmat(
                    os.path.join(params_dir.format(gender), f'{sample_idx:04d}.mat'))['shape'])
                
                data_dict['genders'].append(GENDER_TO_INT_DICT[gender])

                smpl_model = create_model(gender)
                smpl_output = set_shape(smpl_model, data_dict['shapes'][-1][0])
                verts = smpl_output.vertices.detach().cpu().numpy().squeeze()
                faces = smpl_model.faces.squeeze()
                body_mesh = trimesh.Trimesh(vertices=verts, faces=faces, 
                    vertex_colors=np.tile(colors['grey'], (6890, 1)))

                with open(os.path.join(est_kptss_dir.format(gender), f'{sample_idx:04d}_keypoints.json')) as json_f:
                    data_dict['est_kptss'].append(process_openpose(json.load(json_f)))
                data_dict['gt_kptss'].append(smpl_output.joints.detach().cpu().numpy().squeeze())
                data_dict['front_silhs'].append(img_to_silhouette(imread(
                    os.path.join(front_silhss_template.format(gender), f'{sample_idx:04d}.png'))))
                data_dict['side_silhs'].append(img_to_silhouette(imread(
                    os.path.join(side_silhss_template.format(gender), f'{sample_idx:04d}.png'))))
                data_dict['vertss'].append(verts)
                data_dict['facess'].append(faces)
                data_dict['volumes'].append(body_mesh.volume)
            except FileNotFoundError as e:
                print(f'Error with {gender} {sample_idx} ({e.filename})')

        save_dir = save_dir_template.format(gender)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key], dtype=np.float32)
            np.save(os.path.join(save_dir, f'{key}.npy'), data_dict[key])
            data_dict[key] = []


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
        'front_silhs': [],     # too big to load into RAM
        'side_silhs': [],      # too big to load into RAM
        'vertss': [],
        'volumes': []
    }

    for key in data_dict:
        if 'silhs' in key:
            np.save(os.path.join(save_dir, f'{key}.npy'), np.zeros((1, 1)))
        else:
            for subj_dirname in os.listdir(data_dir):
                print(subj_dirname)
                subj_dirpath = os.path.join(data_dir, subj_dirname)

                data = np.load(os.path.join(subj_dirpath, f'{ATTR_MAP[key]}.npy'))
                data_dict[key].append(data)
                # TODO: Use estimated keypoints (OpenPose).

            data_dict[key] = np.array(data_dict[key], dtype=np.float32)
            np.save(os.path.join(save_dir, f'{key}.npy'), data_dict[key])

            data_dict[key] = []


def prepare_star():
    ATTR_MAP = {
        'genders': 'gender',
        'poses': 'pose',
        'shapes': 'shape',
        'vertss': 'verts',
        'volumes': 'volume'
    }

    data_dir = os.path.join('data', 'star', 'generated')
    kpts_dir = os.path.join('data', 'star', 'keypoints')  # for est_kptss
    # TODO: Create directories for both genders.
    save_dir = os.path.join('data', 'star', 'prepared', 'male')

    Path(kpts_dir).mkdir(parents=True, exist_ok=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    data_dict = {
        'genders': [],
        'poses': [],
        'shapes': [],
        'vertss': [],
        'volumes': []
    }

    for key in data_dict:
        for subj_dirname in os.listdir(data_dir):
            print(subj_dirname)
            subj_dirpath = os.path.join(data_dir, subj_dirname)

            data = np.load(os.path.join(subj_dirpath, f'{ATTR_MAP[key]}.npy'))
            data_dict[key].append(data)

        data_dict[key] = np.array(data_dict[key], dtype=np.float32)

        np.save(os.path.join(save_dir, f'{key}.npy'), data_dict[key])

        data_dict[key] = []


def prepare_3dpw():
    data_dir = '/media/kristijan/kristijan-hdd-ex/datasets/3dpw'
    est_kptss_dir = os.path.join(data_dir, 'keypoints', '{}', 'front')
    metadata_dir = os.path.join(data_dir, 'sequenceFiles', 'sequenceFiles', 'test')

    save_dir_template = os.path.join('data', '3dpw', 'prepared', '{}')

    data_dict = {
        'genders': [[], []],
        'poses': [[], []],
        'shapes': [[], []],
        'vertss': [[], []],
        'facess': [[], []],
        'volumes': [[], []]
    }

    for fname in os.listdir(metadata_dir):
        fpath = os.path.join(metadata_dir, fname)
        with open(fpath, 'rb') as fmeta:
            data = pickle.load(fmeta, encoding='latin1')
            
        for subject_idx in range(len(data['genders'])):
            shape = data['betas'][subject_idx][:10]
            #for frame_idx in range(len(data['img_frame_ids'])):
            for frame_idx in range(100):
                print(fname, subject_idx, frame_idx)
                gender = data['genders'][subject_idx]
                gender_idx = 0 if gender == 'm' else 1
                data_dict['genders'].append(0 if gender == 'm' else 1)

                data_dict['poses'][gender_idx].append(data['poses'][subject_idx][frame_idx])
                data_dict['shapes'][gender_idx].append(shape)

                smpl_model = create_model(GENDER_TO_STR_DICT[gender_idx])
                smpl_output = set_shape(smpl_model, shape)
                verts = smpl_output.vertices.detach().cpu().numpy().squeeze()
                faces = smpl_model.faces.squeeze()
                body_mesh = trimesh.Trimesh(vertices=verts, faces=faces, 
                    vertex_colors=np.tile(colors['grey'], (6890, 1)))

                data_dict['vertss'][gender_idx].append(verts)
                data_dict['facess'][gender_idx].append(faces)
                data_dict['volumes'][gender_idx].append(body_mesh.volume)

    for key in data_dict:
        for gender_idx in range(2):
            save_dir = save_dir_template.format(GENDER_TO_STR_DICT[gender_idx])
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            data_dict[key][gender_idx] = np.array(data_dict[key][gender_idx], dtype=np.float32)
            np.save(os.path.join(save_dir, f'{key}.npy'), data_dict[key][gender_idx])
        data_dict[key] = []


if __name__ == '__main__':
    #prepare_caesar()
    #prepare_gt('star')
    #prepare_star()
    prepare_3dpw()
