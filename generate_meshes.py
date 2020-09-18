import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import trimesh
from PIL import Image
from pathlib import Path

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
IMG_DIR = 'imgs/'
GT_DIR = 'gt/'

LOG = False

os.environ['PYOPENGL_PLATFORM'] = 'egl' # Uncommnet this line while running remotely


def render_sample(vertices, faces, subject_idx, pose_idx):
    img_dir = os.path.join(DATA_DIR, IMG_DIR, f'S{subject_idx}')
    # NOTE: dir_idx is equal to view_idx.
    for dir_idx in range(4):
        Path(os.path.join(img_dir, str(dir_idx))).mkdir(parents=True, exist_ok=True)

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


def generate_sample(model, shape_coefs, body_pose, sid, pid):
    model.register_parameter('body_pose', 
            nn.Parameter(body_pose, requires_grad=True))

    inference_time = time.time()
    output = model(betas=shape_coefs, return_verts=True)
    if LOG:
        print(f'Inference time: {time.time() - inference_time}')

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = model.faces.squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    render_start = time.time()
    render_sample(vertices, faces, sid, pid)
    if LOG:
        print(f'Render time: {time.time() - render_start}')

    joint_dir = os.path.join(DATA_DIR, GT_DIR, f'S{sid}')
    Path(joint_dir).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(joint_dir, f'{pid:08d}.npy'), joints)


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


def main(num_poses, num_subjects, num_coefs=10):
    vposer_model, _ = load_vposer(VPOSER_DIR, vp_model='snapshot')

    body_poses  = sample_poses(vposer_model, num_poses)
    shape_coefs = torch.from_numpy(np.random.normal(
        0., 1., size=(num_subjects, 1, 10)).astype(np.float32))

    model = smplx.create(MODELS_DIR, model_type='smplx',
                         gender='neutral', use_face_contour=False,
                         num_betas=num_coefs,
                         body_pose=body_poses[0],
                         ext='npz')

    for subject_idx in range(num_subjects):
        subject_time = time.time()
        for pose_idx in range(num_poses):
            print('S {} P {}'.format(subject_idx, pose_idx))
            sample_time = time.time()
            generate_sample(model, shape_coefs[subject_idx], body_poses[pose_idx],
                    subject_idx, pose_idx)
            if LOG:
                print(f'Sample generation time: {time.time() - sample_time}')
        print(f'Subject generation time: {time.time() - subject_time}')



if __name__ == '__main__':
    all_time = time.time()
    main(num_poses=50000, num_subjects=50)
    if LOG:
        print(f'All time: {time.time() - all_time}')

