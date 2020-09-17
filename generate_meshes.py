import os

import numpy as np
import torch

import smplx
from human_body_prior.tools.model_loader import load_vposer


MODELS_DIR = 'models/'
MESHES_DIR = 'meshes/'
VPOSER_DIR = 'vposer_v1_0'


def store_mesh(path, vertices, faces):
    with open(path, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


# TODO: Pose coefs instead of expression.
def create_mesh(model_dir, mesh_path, shape_coefs, body_pose):
    model = smplx.create(model_dir, model_type='smplx',
                         gender='neutral', use_face_contour=False,
                         num_betas=10,
                         body_pose=body_pose,
                         num_expression_coeffs=10,
                         ext='npz')
    print(model)
    output = model(betas=shape_coefs, return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = model.faces.squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    store_mesh(mesh_path, vertices, faces)
    # TODO: Also store joints.


def generate_meshes():
    vposer_model, _ = load_vposer(VPOSER_DIR, vp_model='snapshot')
    body_pose = vposer_model.sample_poses(num_poses=1)[0][0].reshape(1, -1)
    shape_coefs = torch.randn([1, 10], dtype=torch.float32)
    mesh_path = os.path.join(MESHES_DIR, 'sample.obj')
    create_mesh(MODELS_DIR, mesh_path, shape_coefs, body_pose)


if __name__ == '__main__':
    generate_meshes()

