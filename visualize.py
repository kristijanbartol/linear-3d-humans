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


def visualize_measure_errors(measure_errors):
    fig_path = os.path.join('vis/', f'measurement_errors.png')

    measure_dict = dict(zip(
        MeshMeasurements.alllabels(), 
        [[x] for x in measure_errors])
    )
    pd_params = pd.DataFrame(measure_dict)

    plt.figure()
    sns.set_theme(style="darkgrid")
    fig = sns.barplot(data=pd_params)

    fig.get_figure().savefig(fig_path)


def visualize_feature_importances(model, args):
    fig_path = os.path.join('vis/', f'feature_importances.png')

    feature_importance_dict = dict(zip(
        Regressor.get_labels(args), 
        [[x] for x in Models.feature_importances(model)])
    )
    pd_params = pd.DataFrame(feature_importance_dict)

    plt.figure()
    sns.set_theme(style="darkgrid")
    fig = sns.barplot(data=pd_params)

    fig.get_figure().savefig(fig_path)


# TODO: Convert this function to evaluate means of more than one model.
def visualize_s2s_dists(s2s_dists_array):
    # TODO: Visualize both male and female mesh errors.
    model = create_model('male')
    model_output = set_shape(model, np.zeros((1, 10)))
    vertices = model_output.vertices.detach().cpu().numpy().squeeze()
    faces = model.faces.squeeze()

    imw, imh = 1600, 1600
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    mv.set_background_color(colors['white'])

    overall_max = s2s_dists_array.max()
    for s2s_idx, s2s_dists in enumerate(s2s_dists_array):

        #error_colors = np.array([[x / s2s_dists.max(), 0.5, 0.] for x in s2s_dists])
        error_colors = np.array([[x / overall_max, 0., 1. - x / overall_max] for x in s2s_dists])

        body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, 
                #vertex_colors=np.tile(colors['grey'], (6890, 1)))
                vertex_colors=error_colors)

        mv.set_meshes([body_mesh], group_name='static')
        img = mv.render()

        rgb = Image.fromarray(img, 'RGB')
        rgb.save(os.path.join('vis', f's2s_{s2s_idx}.png'))


def visualize(params_errors, measurement_errors, s2s_dists):
    #visualize_param_errors(params_errors)
    #visualize_measure_errors(measurement_errors)
    #visualize_feature_importances(model, args)
    visualize_s2s_dists(s2s_dists)


def visualize_all():
    # Select male and female indexes to visualize at once.
    # Take all predictions and visualize mean s2s errors.
    pass
