import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
import trimesh
from PIL import Image

from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import colors
from mesh_viewer import MeshViewer

from load import MeshMeasurements, Regressor
from models import Models
from generate import create_model, set_shape


def visualize_param_errors(params_errors):
    fig_path = os.path.join('vis/', f'params_errors.png')

    params_dict = dict(zip(
        [f'PCA{x}' for x in range(params_errors.shape[0])], 
        [[x] for x in params_errors])
    )
    pd_params = pd.DataFrame(params_dict)

    plt.figure()
    sns.set_theme(style="darkgrid")
    fig = sns.barplot(data=pd_params)

    fig.get_figure().savefig(fig_path)


def visualize_measure_errors(measure_errors, method='linear', args=None):
    if args is not None:
        fig_name = f'{method}_measurement_errors_{args.height_noise}_{args.weight_noise}.png' 
    else: 
        fig_name = f'{method}_measurement_errors.png'
    fig_path = os.path.join('vis/', fig_name)

    measure_dict = dict(zip(
        MeshMeasurements.alllabels(), 
        [[x] for x in measure_errors])
    )
    pd_params = pd.DataFrame(measure_dict)

    plt.figure()
    sns.set_theme(style="darkgrid")
    fig = sns.barplot(data=pd_params)

    fig.get_figure().savefig(fig_path)


def visualize_s2s_dists(s2s_dists_array, gender='male', shapes=np.zeros((3, 1, 10)), methods=['linear'], subject_idx=0):
    model = create_model(gender)
    faces = model.faces.squeeze()

    imw, imh = 1600, 1600
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    mv.set_background_color(colors['white'])

    if len(s2s_dists_array.shape) == 1:
        s2s_dists_array = s2s_dists_array.reshape((1, -1))

    overall_max = s2s_dists_array.max()
    for s2s_idx, s2s_dists in enumerate(s2s_dists_array):
        model_output = set_shape(model, shapes[s2s_idx])
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        error_colors = np.array([[x / overall_max, 0., 1. - x / overall_max] for x in s2s_dists])

        body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, 
                #vertex_colors=np.tile(colors['grey'], (6890, 1)))      # use this temporarily for Fig 1 mesh
                vertex_colors=error_colors)

        apply_mesh_tranfsormations_([body_mesh],
                                    trimesh.transformations.rotation_matrix(
                                        np.radians(45), (0, 1, 0)))
        mv.set_meshes([body_mesh], group_name='static')
        img = mv.render()

        rgb = Image.fromarray(img, 'RGB')
        rgb.save(os.path.join('vis', f's2s_{gender}_{methods[s2s_idx]}_{subject_idx}.png'))


def visualize(params_errors, measurement_errors, s2s_dists):
    #visualize_param_errors(params_errors)
    #visualize_measure_errors(measurement_errors)
    #visualize_feature_importances(model, args)
    #visualize_s2s_dists(s2s_dists)
    pass


def visualize_individual():
    # Select male and female indexes to visualize at once.
    SUBJECT_IDXS = [
        [817, 1298],    # male
        [338, 2516]     # female
    ]

    GENDERS = ['male', 'female']
    METHODS = ['expose', 'smplify', 'linear']

    for gender_idx, gender in enumerate(GENDERS):
        for subject_idx in SUBJECT_IDXS[gender_idx]:
            all_params = []
            all_measurement_errors = []
            all_s2s_errors = []
            
            for method in METHODS:
                suffix = '_0.01_1.5' if method == 'linear' else ''

                params = np.load(f'./results/{gender}_{method}_params{suffix}.npy')
                measurement_errors = np.load(f'./results/{gender}_{method}_measurement_errors{suffix}.npy')
                s2s_errors = np.load(f'./results/{gender}_{method}_s2s_errors{suffix}.npy')
                subject_idxs = np.load(f'./results/{gender}_{method}_subject_idxs{suffix}.npy')

                array_idx = np.where(subject_idxs == subject_idx)

                if array_idx[0].shape[0] == 2:
                    print(f'{gender} {subject_idx} {method} fishy!')

                #assert(array_idx[0].shape[0] == 1)
                array_idx = array_idx[0][0]

                all_params.append(params[array_idx].reshape((1, 10)))
                all_measurement_errors.append(measurement_errors[array_idx])
                all_s2s_errors.append(s2s_errors[array_idx])

            all_params = np.array(all_params)
            all_measurement_errors = np.array(all_measurement_errors)
            all_s2s_errors = np.array(all_s2s_errors)

            visualize_s2s_dists(all_s2s_errors, gender, all_params, METHODS, subject_idx)


def visualize_mean():
    GENDERS = ['male', 'female']
    METHODS = ['expose', 'smplify', 'linear']

    for gender_idx, gender in enumerate(GENDERS):
        all_params = []
        all_measurement_errors = []
        all_s2s_errors = []
        
        for method in METHODS:
            suffix = '_0.01_1.5' if method == 'linear' else ''

            measurement_errors = np.load(f'./results/{gender}_{method}_measurement_errors{suffix}.npy')
            s2s_errors = np.load(f'./results/{gender}_{method}_s2s_errors{suffix}.npy')

            all_measurement_errors.append(measurement_errors.mean(axis=0))
            all_s2s_errors.append(s2s_errors.mean(axis=0))

        all_measurement_errors = np.array(all_measurement_errors)   # NOTE: Currently not used
        all_s2s_errors = np.array(all_s2s_errors)

        visualize_s2s_dists(all_s2s_errors, gender=gender, methods=METHODS)


if __name__ == '__main__':
    #visualize_individual()
    visualize_mean()
