import numpy as np
import trimesh
from generate import create_model, set_shape
from PIL import Image

from human_body_prior.tools.omni_tools import colors

from load import MeshMeasurements
from mesh_viewer import MeshViewer

MEAS_COEF_FILE = './results/{}_meas_coefs.npy'
SHAPE_COEF_FILE = './results/{}_shape_coefs.npy'


if __name__ == '__main__':
    gender_in = input('Enter your gender ([male, female]):')
    str_in = input('Enter your height [in meters] and weight [in kg]: ')
    h, w = [float(x) for x in str_in.split()]

    meas_coefs = np.load(MEAS_COEF_FILE.format(gender_in))
    shape_coefs = np.load(SHAPE_COEF_FILE.format(gender_in))

    # NOTE: Demo only works for coefficients without the interaction terms.
    measurements = h * meas_coefs[:, 0] + w * meas_coefs[:, 1] + \
        w / h**2 * meas_coefs[:, 2] + w * h * meas_coefs[:, 3] + meas_coefs[:, 4]
    shape_params = h * shape_coefs[:, 0] + w * shape_coefs[:, 1] + shape_coefs[:, 2]

    for midx, mname in enumerate(MeshMeasurements.aplabels()):
        print(f'{mname}: {measurements[midx] * 100:.2f}cm')

    try:
        model = create_model(gender_in)
        faces = model.faces.squeeze()
        model_output = set_shape(model, shape_params.reshape((1, -1)))
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, 
                vertex_colors=np.tile(colors['grey'], (6890, 1)))

        imw, imh = 800, 800
        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
        mv.set_background_color(colors['black'])

        mv.set_meshes([body_mesh], group_name='static')
        img = mv.render()

        rgb = Image.fromarray(img, 'RGB')
        rgb.show()
    except Exception:
        print('ERROR: Please install packages. :)')
