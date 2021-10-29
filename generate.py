import os
import numpy as np
import torch
import trimesh
from PIL import Image
from pathlib import Path
import argparse

import smplx

from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import colors
from mesh_viewer import MeshViewer
from utils import all_combinations_with_permutations, img_to_silhouette


MODELS_DIR = 'models/'
VPOSER_DIR = 'vposer_v1_0/'
DATA_DIR = 'data/'
IMG_DIR_TEMPLATE = os.path.join(DATA_DIR, '{}/img/')
DATA_DIR_TEMPLATE = os.path.join(DATA_DIR, '{}/generated/')

GENDER_TO_INT_DICT = { 
        'male': 0,
        'female': 1,
        'neutral': 2
    }

GENDER_TO_STR_DICT = { 
        0: 'male',
        1: 'female',
        2: 'neutral'
    }

SMPL_NUM_KPTS = 23
SMPLX_NUM_KPTS = 21


def render_sample(body_mesh, dataset_name, gender, subject_idx):
    img_dir = os.path.join(
            IMG_DIR_TEMPLATE.format(dataset_name), f'{gender}{subject_idx:05d}')
    os.makedirs(os.path.join(img_dir, '0'), exist_ok=True)

    imw, imh = 1600, 1600
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    mv.set_background_color(colors['black'])

    silhouettes = []
    for view, angle in zip(['front', 'side'], [0, 90]):
        apply_mesh_tranfsormations_([body_mesh],
                                    trimesh.transformations.rotation_matrix(
                                        np.radians(angle), (0, 1, 0)))
        mv.set_meshes([body_mesh], group_name='static')
        img = mv.render()

        rgb = Image.fromarray(img, 'RGB')
        rgb_path = os.path.join(img_dir, '0', f'rgb_{view}.png')
        rgb.save(rgb_path)

        silhouettes.append(img_to_silhouette(img))

    return silhouettes


def save(save_dir, gender, shape_coefs):
    np.save(os.path.join(save_dir, f'gender.npy'), GENDER_TO_INT_DICT[gender])
    np.save(os.path.join(save_dir, f'shape.npy'), shape_coefs)


def save_star(save_dir, gender, vertices, faces, shape_coefs, body_pose, volume):
    np.save(os.path.join(save_dir, f'gender.npy'), GENDER_TO_INT_DICT[gender])
    np.save(os.path.join(save_dir, f'verts.npy'), vertices)
    np.save(os.path.join(save_dir, f'faces.npy'), faces)
    np.save(os.path.join(save_dir, f'shape.npy'), shape_coefs)
    np.save(os.path.join(save_dir, f'pose.npy'), body_pose)
    np.save(os.path.join(save_dir, f'volume.npy'), volume)


def set_shape(model, shape_coefs):
    if type(shape_coefs) != torch.Tensor:
        shape_coefs = torch.unsqueeze(torch.tensor(shape_coefs, dtype=torch.float32), dim=0)
        return model(betas=shape_coefs, return_verts=True)
    if type(model) == smplx.star.STAR:
        return model(pose=torch.zeros((1, 72), device='cpu'), betas=shape_coefs, trans=torch.zeros((1, 3), device='cpu'))


def create_model(gender, num_coefs=10, model_type='smpl'):
    if model_type == 'star':
        return smplx.star.STAR()
    else:
        if model_type == 'smpl':
            body_pose = torch.zeros((1, SMPL_NUM_KPTS * 3))
        elif model_type == 'smplx':
            body_pose = torch.zeros((1, SMPLX_NUM_KPTS * 3))
        return smplx.create(MODELS_DIR, model_type=model_type,
                            gender=gender, use_face_contour=False,
                            num_betas=num_coefs,
                            body_pose=body_pose,
                            ext='npz')


def generate_sample(dataset_name, gender, model, shape_coefs, body_pose, sid):
    output = set_shape(model, shape_coefs)

    if type(model) != smplx.star.STAR:
        #joints = output.joints.detach().cpu().numpy().squeeze()

        #vertices = output.vertices.detach().cpu().numpy().squeeze()
        #faces = model.faces.squeeze()
        #body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, 
        #    vertex_colors=np.tile(colors['grey'], (6890, 1)))
        #silhouettes = render_sample(body_mesh, dataset_name, gender, 
        #        sid)
        
        save_dir = os.path.join(DATA_DIR_TEMPLATE.format(dataset_name), f'{gender}{sid:05d}')
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        save(save_dir, gender, shape_coefs)

    else:
        vertices = output.detach().cpu().numpy().squeeze()
        faces = output.f

        body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, 
            vertex_colors=np.tile(colors['grey'], (6890, 1)))

        save_dir = os.path.join(DATA_DIR_TEMPLATE.format(dataset_name), f'{gender}{sid:05d}')
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        save_star(save_dir, gender, vertices, faces, shape_coefs, body_pose, body_mesh.volume)


def generate_subjects(dataset_name, gender, model_type, num_subjects, regenerate=False, num_coefs=10):

    def get_last_idx(dataset_name, gender):
        subject_dirnames = [x for x in os.listdir(
            DATA_DIR_TEMPLATE.format(dataset_name)) if 'npy' not in x \
                and x.startswith(gender)]
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
    np_shape_coefs = np.empty(shape=(num_subjects + start_idx, 1, num_coefs), dtype=np.float32)
    for perm_idx, perm in enumerate(
            list(shape_combination_coefs)[start_idx : start_idx + num_subjects]):
        perm = np.array(perm)
        perm[0] = 0.
        np_shape_coefs[perm_idx + start_idx] = perm
    shape_coefs = torch.from_numpy(np_shape_coefs).to('cpu')
    zero_pose = np.zeros([1, SMPL_NUM_KPTS * 3])
    # NOTE: Generating SMPL-X models.
    model = create_model(gender, model_type=model_type)

    # NOTE: If not regenerate, last shape will still be regenerated, which is OK.
    for subject_idx in range(start_idx, start_idx + num_subjects):
            print('{} {}'.format(gender, subject_idx))
            generate_sample(dataset_name,
                    gender,
                    model, 
                    shape_coefs[subject_idx], 
                    zero_pose,
                    subject_idx
            )


def main(dataset_name, num_neutral=0, num_male=0, 
        num_female=0, model='smpl', regenerate=False, num_coefs=10):
    
    # Create dataset dir and img/ and gt/ subdirs.
    Path(IMG_DIR_TEMPLATE.format(dataset_name)).mkdir(
            parents=True, exist_ok=True)
    os.makedirs(DATA_DIR_TEMPLATE.format(dataset_name), exist_ok=True)
    
    generate_subjects(dataset_name,
            'neutral', 
            model,
            num_neutral, 
            regenerate
    )
    generate_subjects(dataset_name,
            'male', 
            model,
            num_male, 
            regenerate
    )
    generate_subjects(dataset_name,
            'female', 
            model,
            num_female, 
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
            '--model', type=str, help='smpl or star model'
    )
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
    return args


if __name__ == '__main__':
    parser = init_argparse()
    args = check_args(parser.parse_args())
    main(dataset_name=args.name,
         num_neutral=args.neutral,
         num_male=args.male,
         num_female=args.female,
         model=args.model,
         regenerate=args.regenerate)

