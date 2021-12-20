import numpy as np
import trimesh
from generate import create_model, set_shape
from PIL import Image

from human_body_prior.tools.omni_tools import colors

from load import MeshMeasurements
from mesh_viewer import MeshViewer

MEAS_COEF_FILE = './results/{}_meas_coefs.npy'
SHAPE_COEF_FILE = './results/{}_shape_coefs.npy'


def create_mesh_object(gender, shape_params):
    model = create_model(gender)
    faces = model.faces.squeeze()
    model_output = set_shape(model, shape_params.reshape((1, -1)))
    vertices = model_output.vertices.detach().cpu().numpy().squeeze()

    return MeshMeasurements(vertices, faces, keep_mesh=True)


def experiment_weight():
    gender_in = 'male'
    h = 1.72

    meas_coefs = np.load(MEAS_COEF_FILE.format(gender_in))
    shape_coefs = np.load(SHAPE_COEF_FILE.format(gender_in))

    for w in range(50, 90):
        measurements = h * meas_coefs[:, 0] + w * meas_coefs[:, 1] + meas_coefs[:, 2]
        shape_params = h * shape_coefs[:, 0] + w * shape_coefs[:, 1] + shape_coefs[:, 2]

        print(f'{w}\n==============')
        for midx, mname in enumerate(MeshMeasurements.aplabels()):
            print(f'{mname}: {measurements[midx] * 100:.2f}cm')

        print('\n')

        try:
            mesh_object = create_mesh_object(gender_in, shape_params)

            imw, imh = 800, 800
            mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
            mv.set_background_color(colors['black'])

            mv.set_meshes([mesh_object.mesh], group_name='static')
            img = mv.render()

            rgb = Image.fromarray(img, 'RGB')
            #rgb.show()
            rgb.save(f'./vis/weights/{w}.png')
        except Exception:
            print('ERROR: Please install packages. :)')


def experiment_height():
    gender_in = 'male'
    ref_weight = 70
    ref_height = 1.8

    meas_coefs = np.load(MEAS_COEF_FILE.format(gender_in))
    shape_coefs = np.load(SHAPE_COEF_FILE.format(gender_in))

    ref_measurements = ref_height * meas_coefs[:, 0] + ref_weight * meas_coefs[:, 1] + meas_coefs[:, 2]
    ref_shape_params = ref_height * shape_coefs[:, 0] + ref_weight * shape_coefs[:, 1] + shape_coefs[:, 2]

    for pca0 in range(-3000, 3000, 300):
        pca0 /= 1000.
        #h /= 100.
        #w = ref_weight / (h / ref_height)
        w = ref_weight
        #shape_params = h * shape_coefs[:, 0] + w * shape_coefs[:, 1] + shape_coefs[:, 2]
        ref_shape_params[0] = pca0
        mesh_object = create_mesh_object(gender_in, ref_shape_params)
        measurements = mesh_object.apmeasurements

        print(f'{mesh_object.overall_height}m\n==============')

        mesh_object._scale_mesh(ref_height)
        scaled_measurements = mesh_object.apmeasurements

        #print(f'{h}cm\n==============')
        for midx, mname in enumerate(MeshMeasurements.aplabels()):
            print(f'{mname}: {(measurements[midx] - ref_measurements[midx])* 100:.2f}cm '
                f'{(scaled_measurements[midx] - ref_measurements[midx])* 100:.2f}cm')
        print(ref_shape_params)

        print('\n')

        imw, imh = 800, 800
        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
        mv.set_background_color(colors['black'])

        mv.set_meshes([mesh_object.mesh], group_name='static')
        img = mv.render()

        rgb = Image.fromarray(img, 'RGB')
        #rgb.show()
        rgb.save(f'./vis/heights/pca0_{pca0}.png')


if __name__ == '__main__':
    experiment_height()
