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

import smplx

from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import colors, makepath
from human_body_prior.mesh.mesh_viewer import MeshViewer
from human_body_prior.tools.visualization_tools import imagearray2file, smpl_params2ply


MODELS_DIR = 'models/'
VPOSER_DIR = 'vposer_v1_0/'
DATA_DIR = 'data/'
IMG_DIR_TEMPLATE = os.path.join(DATA_DIR, '{}/imgs/')
GT_DIR_TEMPLATE = os.path.join(DATA_DIR, '{}/gt/')

os.environ['PYOPENGL_PLATFORM'] = 'egl' # Uncommnet this line while running remotely


def render_sample(vertices, faces, dataset_name, subject_idx, pose_idx):
#    img_dir = os.path.join(DATA_DIR, IMG_DIR, f'S{subject_idx}')
    img_dir = os.path.join(
            IMG_DIR_TEMPLATE.format(dataset_name), f'S{subject_idx}')
    for view_idx in range(4):
        os.makedirs(os.path.join(img_dir, str(view_idx)), exist_ok=True)
#        Path(os.path.join(img_dir, str(view_idx))).mkdir(
#                parents=True, exist_ok=True)

    view_angles = [0, 90, 180, 270]
    imw, imh = 600, 600
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, 
            vertex_colors=np.tile(colors['grey'], (6890, 1)))
    for view_idx, angle in enumerate(view_angles):
        apply_mesh_tranfsormations_([body_mesh],
                                    trimesh.transformations.rotation_matrix(
                                        np.radians(angle), (0, 1, 0)))
        mv.set_meshes([body_mesh], group_name='static')

        image = mv.render()
        image = Image.fromarray(image, 'RGB')
        img_path = os.path.join(img_dir, str(view_idx), f'{pose_idx:08d}.png')
        image.save(img_path)

        apply_mesh_tranfsormations_([body_mesh],
                                    trimesh.transformations.rotation_matrix(
                                        np.radians(-angle), (0, 1, 0)))

def save_joints(joints, dataset_name, subject_idx, pose_idx):
    joint_dir = os.path.join(GT_DIR_TEMPLATE.format(dataset_name), f'S{subject_idx}')
    Path(joint_dir).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(joint_dir, f'{pose_idx:08d}.npy'), joints)


def generate_sample(dataset_name, model, shape_coefs, body_pose, sid, pid):
    model.register_parameter('body_pose', 
            nn.Parameter(body_pose, requires_grad=True))

    output = model(betas=shape_coefs, return_verts=True)

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = model.faces.squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    render_sample(vertices, faces, dataset_name, sid, pid)
    save_joints(joints, dataset_name, sid, pid)


def create_model(gender, init_body_pose, num_coefs=10):
    return smplx.create(MODELS_DIR, model_type='smplx',
                        gender=gender, use_face_contour=False,
                        num_betas=num_coefs,
                        body_pose=init_body_pose,
                        ext='npz')


def save_params(dataset_name, sid, shape_coefs, gender):
    save_dir = os.path.join(GT_DIR_TEMPLATE.format(dataset_name), f'S{sid}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'params.json')
    params = {'betas': shape_coefs[sid].tolist(), 'gender': gender}
    with open(save_path, 'w') as fjson:
        json.dump(params, fjson)


def generate_subjects(dataset_name, gender, start_idx, num_subjects, 
        body_poses, vposer, num_coefs=10):
    if num_subjects <= 0:
        return
    model = create_model(gender, body_poses[0])
    np_coefs = np.random.normal(
            0., 1., size=(num_subjects, 1, num_coefs)).astype(np.float32)
    shape_coefs = torch.from_numpy(np_coefs)

    for subject_idx in range(start_idx, start_idx + num_subjects):
        save_params(dataset_name, subject_idx, np_coefs, gender)
        for pose_idx in range(body_poses.shape[0]):
            print('S {} P {}'.format(subject_idx, pose_idx))
            generate_sample(dataset_name,
                    model, 
                    shape_coefs[subject_idx - start_idx], 
                    body_poses[pose_idx],
                    subject_idx, 
                    pose_idx
            )


def sample_poses(vposer_model, num_poses, output_type='aa'):
    dtype = vposer_model.bodyprior_dec_fc1.weight.dtype
    device = vposer_model.bodyprior_dec_fc1.weight.device
    vposer_model.eval()
    with torch.no_grad():
        Zgen = torch.tensor(np.random.normal(0., 1., 
            size=(num_poses, vposer_model.latentD)), dtype=dtype).to(device)
    body_poses = vposer_model.decode(Zgen, output_type=output_type)
    body_poses = body_poses.reshape(num_poses, 1, -1)
    return body_poses


def main(dataset_name, num_poses, num_neutral=0, num_male=0, 
        num_female=0, num_coefs=10):
    vposer_model, _ = load_vposer(VPOSER_DIR, vp_model='snapshot')
    body_poses  = sample_poses(vposer_model, num_poses)
    
    # Create dataset dir and img/ and gt/ subdirs.
    Path(IMG_DIR_TEMPLATE.format(dataset_name)).mkdir(
            parents=True, exist_ok=True)
    os.makedirs(GT_DIR_TEMPLATE.format(dataset_name), exist_ok=True)

    # Save body pose params (one file per dataset)..
    np.save(os.path.join(GT_DIR_TEMPLATE.format(dataset_name), 'poses.npy'), 
            body_poses.cpu().detach().numpy())
    
    generate_subjects(dataset_name,
            'neutral', 
            0, 
            num_neutral, 
            body_poses, 
            vposer_model
    )
    generate_subjects(dataset_name,
            'male', 
            num_neutral, 
            num_male, 
            body_poses, 
            vposer_model
    )
    generate_subjects(dataset_name,
            'female', 
            num_neutral + num_male, 
            num_female, 
            body_poses, 
            vposer_model
    )


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
            usage='%(prog)s [OPTIONS]...',
            description='Generates specified number of meshes \
            (subject x pose).'
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
         num_female=args.female)

