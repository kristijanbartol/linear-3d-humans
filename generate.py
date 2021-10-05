import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import trimesh
from PIL import Image
from pathlib import Path
import argparse
import json
import pyrender
import random
import itertools

import smplx

from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import colors, makepath
#from human_body_prior.mesh.mesh_viewer import MeshViewer
from mesh_viewer import MeshViewer
from human_body_prior.tools.visualization_tools import imagearray2file, smpl_params2ply
from utils import all_combinations_with_permutations


MODELS_DIR = 'models/'
VPOSER_DIR = 'vposer_v1_0/'
DATA_DIR = 'data/'
IMG_DIR_TEMPLATE = os.path.join(DATA_DIR, '{}/imgs/')
GT_DIR_TEMPLATE = os.path.join(DATA_DIR, '{}/gt/')

GENDER_DICT = { 
        'male': 0,
        'female': 1,
        'neutral': 2
        }

#os.environ['PYOPENGL_PLATFORM'] = 'egl' # Uncommnet this line while running remotely


def img_to_silhouette(img):
    img = img.copy()
    img.setflags(write=1)
    img[img == 255] = 0
    img[img > 0] = 255
    
    return img


def render_sample(body_mesh, dataset_name, gender, subject_idx, pose_idx):
    img_dir = os.path.join(
            IMG_DIR_TEMPLATE.format(dataset_name), f'{gender}{subject_idx:04d}')
    os.makedirs(os.path.join(img_dir, '0'), exist_ok=True)

    rot_step = random.randint(0, 360)
    imw, imh = 2000, 2000
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    mv.set_background_color(colors['black'])

    #apply_mesh_tranfsormations_([body_mesh],
    #                            trimesh.transformations.rotation_matrix(
    #                                np.radians(rot_step), (0, 1, 0)))
    mv.set_meshes([body_mesh], group_name='static')

    img = mv.render()

    rgb = Image.fromarray(img, 'RGB')
    bw = img_to_silhouette(img)
    bw = Image.fromarray(bw, 'RGB')
    
    rgb_path = os.path.join(img_dir, '0', f'rgb_{pose_idx:06d}.png')
    bw_path = os.path.join(img_dir, '0', f'silhouette_{pose_idx:06d}.png')
    
    rgb.save(rgb_path)
    bw.save(bw_path)

    # TODO: Generate at least two silhouettes (front, side).
    return [rgb], [bw]


def save(save_dir, pid, joints, vertices, faces, shape_coefs, body_pose, volume):
    np.save(os.path.join(save_dir, f'joints_{pid:06d}.npy'), joints)
    np.save(os.path.join(save_dir, f'verts_{pid:06d}.npy'), vertices)
    np.save(os.path.join(save_dir, f'faces_{pid:06d}.npy'), faces)
    np.save(os.path.join(save_dir, f'shape.npy'), shape_coefs)
    np.save(os.path.join(save_dir, f'pose_{pid:06d}.npy'), body_pose)
    np.save(os.path.join(save_dir, f'volume.npy'), volume)


def generate_sample(dataset_name, gender, model, shape_coefs, body_pose, 
        sid, pid):
    model.register_parameter('body_pose', 
            nn.Parameter(body_pose, requires_grad=False))

    output = model(betas=shape_coefs, return_verts=True)

    joints = output.joints.detach().cpu().numpy().squeeze()

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = model.faces.squeeze()
    body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, 
        vertex_colors=np.tile(colors['grey'], (6890, 1)))
    images, silhouettes = render_sample(body_mesh, dataset_name, gender, 
            sid, pid)
    
    save_dir = os.path.join(GT_DIR_TEMPLATE.format(dataset_name), f'{gender}{sid:04d}')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    save(save_dir, pid, joints, vertices, faces, shape_coefs, body_pose, body_mesh.volume)


def create_model(gender, init_body_pose, num_coefs=10):
    return smplx.create(MODELS_DIR, model_type='smplx',
                        gender=gender, use_face_contour=False,
                        num_betas=num_coefs,
                        body_pose=init_body_pose,
                        ext='npz')


def generate_subjects(dataset_name, gender, num_subjects, 
        body_poses, vposer, regenerate=False, num_coefs=10):

    def get_last_idx(dataset_name, gender):
        subject_dirnames = [x for x in os.listdir(
            GT_DIR_TEMPLATE.format(dataset_name)) if 'npy' not in x]
        # IF the dataset is newly created.
        if len(subject_dirnames) == 0:
            return 0
        last_idx = int(sorted(subject_dirnames)[-1][-4:])
        return last_idx

    if regenerate:
        start_idx = 0
    else:
        start_idx = get_last_idx(dataset_name, gender)

    if num_subjects <= 0:
        return

    shape_combination_coefs = all_combinations_with_permutations([0.0, 0.4, 0.8], num_coefs)
    num_subjects = min(num_subjects, len(shape_combination_coefs))
    #np_shape_coefs = np.random.normal(0., 1., 
    #        size=(num_subjects + start_idx, 1, num_coefs)).astype(np.float32)
    np_shape_coefs = np.empty(shape=(num_subjects + start_idx, 1, num_coefs), dtype=np.float32)
    for perm_idx, perm in enumerate(
            list(shape_combination_coefs)[start_idx : start_idx + num_subjects]):
        perm = np.array(perm)
        perm[0] = 0.
        np_shape_coefs[perm_idx + start_idx] = perm
    shape_coefs = torch.from_numpy(np_shape_coefs)
    model = create_model(gender, body_poses[0])

    # NOTE: If not regenerate, last shape will still be regenerated, which is OK.
    for subject_idx in range(start_idx, start_idx + num_subjects):
        # TODO: Move all savings inside generate_sample function.
        #save_params(dataset_name, gender, subject_idx, np_shape_coefs)
        for pose_idx in range(body_poses.shape[0]):
            print('{} {} pose {}'.format(gender, subject_idx, pose_idx))
            generate_sample(dataset_name,
                    gender,
                    model, 
                    shape_coefs[subject_idx], 
                    body_poses[pose_idx],
                    subject_idx, 
                    pose_idx
            )


def sample_poses(vposer_model, num_poses, loaded_np=None, output_type='aa'):
    dtype = vposer_model.bodyprior_dec_fc1.weight.dtype
    device = vposer_model.bodyprior_dec_fc1.weight.device
    vposer_model.eval()
    if loaded_np is not None:
        return torch.tensor(loaded_np, dtype=dtype).to(device)
    with torch.no_grad():
        #Zgen = torch.tensor(np.random.normal(0., 1., 
        #    size=(num_poses, vposer_model.latentD)), dtype=dtype).to(device)
        Zgen = torch.tensor(np.zeros(shape=(num_poses, vposer_model.latentD)), 
            dtype=dtype).to(device)
    body_poses = vposer_model.decode(Zgen, output_type=output_type)
    body_poses = body_poses.reshape(num_poses, 1, -1)

    body_poses = torch.zeros(size=(1, 1, body_poses.shape[2]))

    return body_poses


def main(dataset_name, num_poses, num_neutral=0, num_male=0, 
        num_female=0, regenerate=False, num_coefs=10):
    vposer_model, _ = load_vposer(VPOSER_DIR, vp_model='snapshot')
    
    '''
    poses_path = os.path.join(GT_DIR_TEMPLATE.format(dataset_name), 'poses.npy')
    create_poses_flag = regenerate or (not os.path.exists(poses_path))
    # If poses are already generated, to not override them.
    if create_poses_flag:
        # NOTE: For now, using only neutral pose.
        body_poses  = sample_poses(vposer_model, num_poses)
        gt_dir = GT_DIR_TEMPLATE.format(dataset_name)
        Path(gt_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(gt_dir, 'poses.npy'), body_poses.cpu().detach().numpy())
    else:
        body_poses = torch.tensor(np.load(poses_path))
    '''
    body_poses = torch.zeros(size=(1, 1, vposer_model.num_joints * 3))
    
    # Create dataset dir and img/ and gt/ subdirs.
    Path(IMG_DIR_TEMPLATE.format(dataset_name)).mkdir(
            parents=True, exist_ok=True)
    os.makedirs(GT_DIR_TEMPLATE.format(dataset_name), exist_ok=True)
    
    generate_subjects(dataset_name,
            'neutral', 
            num_neutral, 
            body_poses, 
            vposer_model,
            regenerate
    )
    generate_subjects(dataset_name,
            'male', 
            num_male, 
            body_poses, 
            vposer_model,
            regenerate
    )
    generate_subjects(dataset_name,
            'female', 
            num_female, 
            body_poses, 
            vposer_model,
            regenerate
    )


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
            usage='%(prog)s [OPTIONS]...',
            description='Generates specified number of meshes \
            (subject x pose). If --regenerate, then num_poses \
            is redundant.'
            )
    parser.add_argument(
            '--name', type=str, help='dataset name')
    parser.add_argument(
            '--num_poses', type=int, default=0,
            help='# of poses per subject')
    parser.add_argument(
            '--neutral', type=int, default=0,
            help='# of neutral gender subjects')
    parser.add_argument(
            '--male', type=int, default=0,
            help='# of male subjects')
    parser.add_argument(
            '--female', type=int, default=0,
            help='# of female subjects')
    parser.add_argument(
            '--regenerate', dest='regenerate',
            action='store_true', help='regenerate pose params')
    parser.add_argument(
            '--zero_pose', dest='zero_pose',
            action='store_true', help='generate fixed, zero (neutral) poses')

    return parser


def check_args(args):
    if args.neutral <= 0 and args.male <= 0 and \
            args.female <= 0:
        raise Exception('Should set at least one gender!')
    if args.num_poses <= 0:
        raise Exception('Should generate more than 0 pose!')
    return args


if __name__ == '__main__':
    parser = init_argparse()
    args = check_args(parser.parse_args())
    main(dataset_name=args.name,
         num_poses=args.num_poses, 
         num_neutral=args.neutral,
         num_male=args.male,
         num_female=args.female,
         regenerate=args.regenerate)

