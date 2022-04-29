from math import floor
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

from load import MeshMeasurements
from models import Models
from generate import create_model, set_shape


os.environ['PYOPENGL_PLATFORM'] = 'egl'


all_to_ap_measurement_idxs = [10, 15, 20, 6, 23, 18, 25, 4, 8, 1, 13, 21, 5, 0, 19]


def visualize_measure_errors(measure_errors, label, noise_stds):
    fig_name = f'noisy_measurement_errors_{label}.png'
    fig_path = os.path.join('vis/', fig_name)

    ap_measurement_errors = measure_errors[:, all_to_ap_measurement_idxs].mean(axis=2) * 100.

    noise_stds = [x * 100 if label == 'height' else x for x in noise_stds]
    to_string = lambda x: str(x) + 'cm' if label == 'height' else str(x) + 'kg' 
    labels = ['Measurement'] + [to_string(x) for x in noise_stds]
    data = [np.array(MeshMeasurements.letterlabels())] + [x for x in ap_measurement_errors]
    measure_dict = dict(zip(
        labels, 
        data)
    )
    pd_params = pd.DataFrame(measure_dict)
    pd_params.plot(x='Measurement', y=labels[1:], kind='bar', figsize=(12, 5))

    plt.savefig(fig_path)


def crop_to_content(img):
    for row_idx1 in range(img.shape[0]):
        if not np.all(img[row_idx1] == 255):
            break

    for col_idx1 in range(img.shape[0]):
        if not np.all(img[:, col_idx1] == 255):
            break

    for row_idx2 in range(img.shape[0] - 1, 0, -1):
        if not np.all(img[row_idx2] == 255):
            break

    for col_idx2 in range(img.shape[0] - 1, 0, -1):
        if not np.all(img[:, col_idx2] == 255):
            break

    return img[row_idx1:row_idx2, col_idx1:col_idx2]


def visualize_s2s_dists(s2s_dists_array, gender='male', shapes=np.zeros((3, 1, 10)), methods=['linear'], subject_idx=0):
    model = create_model(gender)
    faces = model.faces.squeeze()

    imw, imh = 1600, 1600
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    mv.set_background_color(colors['white'])

    if len(s2s_dists_array.shape) == 1:
        s2s_dists_array = s2s_dists_array.reshape((1, -1))

    overall_max = s2s_dists_array.max()
    overall_min = s2s_dists_array.min()
    for s2s_idx, s2s_dists in enumerate(s2s_dists_array):
        model_output = set_shape(model, shapes[s2s_idx])
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        #cmap_colors = plt.cm.get_cmap('cool').colors
        cmap = plt.cm.get_cmap('turbo')
        cmap_colors = cmap(np.arange(0, cmap.N))[:, :3]
        color_idxs = np.array([floor(x / overall_max * 256. - 0.1) for x in s2s_dists])
        error_colors = np.array([cmap_colors[x] for x in color_idxs])

        body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, 
                #vertex_colors=np.tile(colors['grey'], (6890, 1)))      # use this temporarily for Fig 1 mesh
                vertex_colors=error_colors)

        apply_mesh_tranfsormations_([body_mesh],
                                    trimesh.transformations.rotation_matrix(
                                        np.radians(45), (0, 1, 0)))
        mv.set_meshes([body_mesh], group_name='static')
        img = mv.render()

        cropped_img = crop_to_content(img)
        rgb = Image.fromarray(cropped_img, 'RGB')
        rgb.save(os.path.join('vis', f's2s_{gender}_{methods[s2s_idx]}_{subject_idx}.png'))

    print(f'{gender} ({subject_idx}) max: {overall_max * 1000.}')
    print(f'{gender} ({subject_idx}) min: {overall_min * 1000.}')


def visualize(params_errors, measurement_errors, s2s_dists):
    #visualize_param_errors(params_errors)
    #visualize_measure_errors(measurement_errors)
    #visualize_feature_importances(model, args)
    #visualize_s2s_dists(s2s_dists)
    pass


def visualize_individual():
    # Select male and female indexes to visualize at once.
    SUBJECT_IDXS = [
        [284, 1298],    # male
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
                suffix = '_0.01_1.5_2' if method == 'linear' else ''
                #suffix += '_2' if gender == 'male' and method == 'linear' else ''

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
            suffix = '_0.01_1.5_2' if method == 'linear' else ''

            measurement_errors = np.load(f'./results/{gender}_{method}_measurement_errors{suffix}.npy')
            s2s_errors = np.load(f'./results/{gender}_{method}_s2s_errors{suffix}.npy')

            all_measurement_errors.append(measurement_errors.mean(axis=0))
            all_s2s_errors.append(s2s_errors.mean(axis=0))

        all_measurement_errors = np.array(all_measurement_errors)   # NOTE: Currently not used
        all_s2s_errors = np.array(all_s2s_errors)

        visualize_s2s_dists(all_s2s_errors, gender=gender, methods=METHODS)


def visualize_noisy_measurements(label, noise_stds):
    fpath_template = './results/male_linear_measurement_errors_{}_{}.npy'

    all_measure_errors = []
    for std in noise_stds:
        kwargs = (std, 0.0) if label == 'height' else (0.0, std)
        all_measure_errors.append(np.load(fpath_template.format(kwargs[0], kwargs[1])))
    
    all_measure_errors = np.array(all_measure_errors)
    visualize_measure_errors(all_measure_errors, label, noise_stds)


def visualize_measurement_distribution(gender, X, all_measurements):
    fig_name = f'bodyfit_distributions_{gender}.png'
    fig_path = os.path.join('vis/', fig_name)

    ap_measurements = all_measurements[:, all_to_ap_measurement_idxs]

    labels = ['Height', 'Weight'] + MeshMeasurements.letterlabels()
    data = np.concatenate([X, ap_measurements], axis=1).swapaxes(0, 1)
    data[2:] *= 100.
    data[0, :] *= 100.
    measure_dict = dict(zip(
        labels, 
        data)
    )
    df = pd.DataFrame(measure_dict)
    df.boxplot(grid=False, figsize=(12, 5), rot=0)
    
    plt.title('BODY-fit+W Males')
    plt.xlabel('Measurement Label')
    plt.ylabel('cm / kg')

    plt.savefig(fig_path)


if __name__ == '__main__':
    visualize_individual()
    visualize_mean()

    #visualize_noisy_measurements('height', [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05])
    #visualize_noisy_measurements('weight', [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
