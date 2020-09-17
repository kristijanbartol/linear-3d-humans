import os
import time
import numpy as np
import torch
import trimesh
from PIL import Image

import smplx

from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import colors, makepath
from human_body_prior.mesh.mesh_viewer import MeshViewer
from human_body_prior.tools.visualization_tools import imagearray2file, smpl_params2ply


MODELS_DIR = 'models/'
MESHES_DIR = 'meshes/'
VPOSER_DIR = 'vposer_v1_0'

os.environ['PYOPENGL_PLATFORM'] = 'egl' # Uncommnet this line while running remotely


def render_sample(vertices, faces):
    view_angles = [0, 90, 180, 270]
    imw, imh = 400, 400
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
        image.save('render{}.png'.format(view_idx))
        apply_mesh_tranfsormations_([body_mesh],
                                    trimesh.transformations.rotation_matrix(
                                        np.radians(-angle), (0, 1, 0)))


def create_mesh(model_dir, mesh_path, shape_coefs, body_pose):
    model_time = time.time()
    model = smplx.create(model_dir, model_type='smplx',
                         gender='neutral', use_face_contour=False,
                         num_betas=10,
                         body_pose=body_pose,
                         num_expression_coeffs=10,
                         ext='npz')
    print('Model creation time: {}'.format(time.time() - model_time))
    output = model(betas=shape_coefs, return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = model.faces.squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    render_start = time.time()
    render_sample(vertices, faces)
    print('Render time: {}'.format(time.time() - render_start))
    # TODO: Also store joints.


def generate_meshes():
    vposer_model, _ = load_vposer(VPOSER_DIR, vp_model='snapshot')
    body_pose = vposer_model.sample_poses(num_poses=1)[0][0].reshape(1, -1)
    shape_coefs = torch.randn([1, 10], dtype=torch.float32)
    mesh_path = os.path.join(MESHES_DIR, 'sample.obj')
    create_mesh(MODELS_DIR, mesh_path, shape_coefs, body_pose)


if __name__ == '__main__':
    all_time = time.time()
    generate_meshes()
    print('All time: {}'.format(time.time() - all_time))

