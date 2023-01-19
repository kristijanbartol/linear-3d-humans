import os
import numpy as np

from .measures import MeshMeasurements
from .generate import GENDER_TO_INT_DICT


def add_interaction_terms(samples_in, num_interaction):
    interaction_terms = np.array([
        samples_in[:, 1] / samples_in[:, 0] ** 2,       # w / h ** 2
        samples_in[:, 1] * samples_in[:, 0],            # w * h
        samples_in[:, 1] ** 2,                          # w ** 2
        samples_in[:, 0] ** 2,                          # h ** 2
        samples_in[:, 1] ** 2 * samples_in[:, 0] ** 2   # w ** 2 * h ** 2
    ][:num_interaction]).swapaxes(0, 1)

    return np.concatenate([samples_in, interaction_terms], axis=1)


def add_noise(samples_in, h_std, w_std):
    samples_in[:, 0] += np.random.normal(0, h_std, samples_in.shape[0])
    samples_in[:, 1] += np.random.normal(0, w_std, samples_in.shape[0])
    return samples_in


def load(args):
    data_dir = os.path.join(args.data_root, args.dataset_name, 'prepared', args.gender)

    regressor_name = 'inputs.npy'
    regressor_path = os.path.join(data_dir, regressor_name)

    data_dict = {}
    for fname in os.listdir(data_dir):
        data_dict[fname.split('.')[0]] = np.load(os.path.join(data_dir, fname))

    if not os.path.exists(regressor_path):
        samples_in = []
        measurements_all = []

        for sample_idx in range(data_dict['genders'].shape[0]):
            verts = data_dict['vertss'][sample_idx]
            faces = data_dict['facess'][sample_idx]
            
            mesh_measurements = MeshMeasurements.from_data(
                gender=args.gender,
                verts=verts, 
                faces=faces, 
                auto_flush=False
            )

            samples_in.append(np.array([
                mesh_measurements.height,
                1000 * mesh_measurements.weight
            ]))
            measurements_all.append(mesh_measurements.all)

        samples_in = np.array(samples_in)
        measurements_all = np.array(measurements_all)

        np.save(regressor_path, samples_in)
        np.save(os.path.join(data_dir, 'measurements.npy'), measurements_all)
    else:
        samples_in = np.load(regressor_path)
        measurements_all = np.load(os.path.join(data_dir, 'measurements.npy'))

    samples_in = add_noise(samples_in, args.height_noise, args.weight_noise)
    
    if args.num_interaction > 0:
        samples_in = add_interaction_terms(samples_in, args.num_interaction)

    return samples_in, data_dict['shapes'][:, 0], measurements_all, data_dict['genders']


def load_from_shapes(args):
    data_dir = os.path.join(
        args.data_root, 
        args.dataset_name, 
        'prepared', 
        args.gender
    )

    suffix = f'_{args.num_interaction}' if args.num_interaction > 0 else ''

    regressor_name = f'inputs_{args.height_noise}_{args.weight_noise}{suffix}.npy'
    regressor_path = os.path.join(data_dir, regressor_name)

    shapes_params = np.load(os.path.join(data_dir, 'shapes.npy'))

    if not os.path.exists(regressor_path):
        samples_in = []
        measurements_all = []

        for shape_idx, shape_params in enumerate(shapes_params):
            if shape_idx % 1000 == 0 and shape_idx != 0:
                print(shape_idx)
                
            mesh_measurements = MeshMeasurements.from_shape_params(
                gender=args.gender, 
                shape=shape_params, 
                auto_flush=True
            )
            
            samples_in.append(np.array([
                mesh_measurements.height,
                1000 * mesh_measurements.weight
            ]))
            measurements_all.append(mesh_measurements.all)

        samples_in = np.array(samples_in)
        measurements_all = np.array(measurements_all)

        np.save(regressor_path, samples_in)
        np.save(os.path.join(data_dir, 'measurements.npy'), measurements_all)
    else:
        samples_in = np.load(regressor_path)
        measurements_all = np.load(os.path.join(data_dir, 'measurements.npy'))
        
    samples_in = add_noise(samples_in, args.height_noise, args.weight_noise)
    
    if args.num_interaction > 0:
        samples_in = add_interaction_terms(samples_in, args.num_interatction)

    genders_all = np.array([GENDER_TO_INT_DICT[args.gender]] * samples_in.shape[0])

    # TODO: Update code (generate.py) so that you remove this odd indexing of shapes.
    return samples_in, shapes_params[:, 0], measurements_all, genders_all
