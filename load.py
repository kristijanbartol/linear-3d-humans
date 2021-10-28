import os
import math
import numpy as np
from pyrender.mesh import Mesh
from sklearn.preprocessing import normalize
import trimesh

from human_body_prior.tools.omni_tools import colors
from mesh_viewer import MeshViewer
from PIL import Image


HORIZ_NORMAL = np.array([0, 1, 0], dtype=np.float32)
VERT_NORMAL = np.array([1, 0, 0], dtype=np.float32)

GENDER_HEIGHT = {
    0: 1.82,    # male
    1: 1.68,    # female
    2: 1.75     # neutral
}


def get_dist(*vs):
    # NOTE: Works both for 3D and 2D joint coordinates.
    total_dist = 0
    for vidx in range(len(vs) - 1):
        total_dist += np.linalg.norm(vs[vidx] - vs[vidx + 1])
    return total_dist


def get_height(v1, v2):
    # NOTE: Works both for 3D and 2D joint coordinates.
    return np.abs((v1 - v2))[1]


class MeshMeasurements:

    # Mesh landmark indexes.
    HEAD_TOP = 412
    LEFT_HEEL = 3463
    LEFT_NIPPLE = 598
    BELLY_BUTTON = 3500
    INSEAM_POINT = 3149
    LEFT_SHOULDER = 3011
    RIGHT_SHOULDER = 6470
    LEFT_CHEST = 1423
    RIGHT_CHEST = 4896
    LEFT_WAIST = 631
    RIGHT_WAIST = 4424
    UPPER_BELLY_POINT = 3504
    REVERSE_BELLY_POINT = 3502
    LEFT_HIP = 1229
    RIGHT_HIP = 4949
    LEFT_MID_FINGER = 2445
    RIGHT_MID_FINGER = 5906
    LEFT_WRIST = 2241
    RIGHT_WRIST = 5702
    LEFT_INNER_ELBOW = 1663
    RIGHT_INNER_ELBOW = 5121

    SHOULDER_TOP = 3068
    LOW_LEFT_HIP = 3134
    LEFT_ANKLE = 3334

    LOWER_BELLY_POINT = 1769
    FOREHEAD_POINT = 335
    NECK_POINT = 3050
    PELVIS_POINT = 3145
    RIGHT_BICEP_POINT = 6281
    RIGHT_FOREARM_POINT = 5084
    RIGHT_THIGH_POINT = 4971
    RIGHT_CALF_POINT = 4589
    RIGHT_ANKLE_POINT = 6723

    # Mesh measurement idnexes.
    OVERALL_HEIGHT = (HEAD_TOP, LEFT_HEEL)
    SHOULDER_TO_CROTCH_HEIGHT = (SHOULDER_TOP, INSEAM_POINT)
    NIPPLE_HEIGHT = (LEFT_NIPPLE, LEFT_HEEL)
    NAVEL_HEIGHT = (BELLY_BUTTON, LEFT_HEEL)
    INSEAM_HEIGHT = (INSEAM_POINT, LEFT_HEEL)

    SHOULDER_BREADTH = (LEFT_SHOULDER, RIGHT_SHOULDER)
    CHEST_WIDTH = (LEFT_CHEST, RIGHT_CHEST)
    WAIST_WIDTH = (LEFT_WAIST, RIGHT_WAIST)
    TORSO_DEPTH = (UPPER_BELLY_POINT, REVERSE_BELLY_POINT)
    HIP_WIDTH = (LEFT_HIP, RIGHT_HIP)

    ARM_SPAN_FINGERS = (LEFT_MID_FINGER, RIGHT_MID_FINGER)
    ARM_SPAN_WRIST = (LEFT_WRIST, RIGHT_WRIST)
    ARM_LENGTH = (LEFT_SHOULDER, LEFT_WRIST)
    FOREARM_LENGTH = (LEFT_INNER_ELBOW, LEFT_WRIST)
    INSIDE_LEG_LENGTH = (LOW_LEFT_HIP, LEFT_ANKLE)

    def __init__(self, gender: int, verts, faces, noise_std=1.5):
        self.gender = gender
        self.verts = verts
        self.faces = faces

        '''
        imw, imh = 1600, 1600
        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
        mv.set_background_color(colors['black'])

        pre_scale_mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces,
            vertex_colors=np.tile(colors['grey'], (6890, 1)))

        mv.set_meshes([pre_scale_mesh], group_name='static')
        img = mv.render()

        rgb = Image.fromarray(img, 'RGB')
        rgb.save('pre-scale.png')
        '''

        #self.verts = self.__scale_vertices(verts)
        self.verts *= (GENDER_HEIGHT[self.gender] / self.overall_height)

        self.mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)


        '''
        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
        mv.set_background_color(colors['black'])

        post_scale_mesh = trimesh.Trimesh(vertices=self.verts, faces=faces, 
            vertex_colors=np.tile(colors['grey'], (6890, 1)))

        mv.set_meshes([post_scale_mesh], group_name='static')
        img = mv.render()

        rgb = Image.fromarray(img, 'RGB')
        rgb.save('post-scale.png')
        '''


        self.volume = self.mesh.volume
        self.weight = (1000 * self.volume) + np.random.normal(0, noise_std)

    @property
    def overall_height(self):
        return get_height(
            self.verts[self.OVERALL_HEIGHT[0]], 
            self.verts[self.OVERALL_HEIGHT[1]]
        )

    @property
    def shoulder_to_crotch(self):
        return get_height(
            self.verts[self.SHOULDER_TO_CROTCH_HEIGHT[0]],
            self.verts[self.SHOULDER_TO_CROTCH_HEIGHT[1]]
        )

    @property
    def nipple_height(self):
        return get_height(
            self.verts[self.NIPPLE_HEIGHT[0]], 
            self.verts[self.NIPPLE_HEIGHT[1]]
        )

    @property
    def navel_height(self):
        return get_height(
            self.verts[self.NAVEL_HEIGHT[0]], 
            self.verts[self.NAVEL_HEIGHT[1]]
        )

    @property
    def inseam_height(self):
        return get_height(
            self.verts[self.INSEAM_HEIGHT[0]], 
            self.verts[self.INSEAM_HEIGHT[1]]
        )

    @property
    def shoulder_breadth(self):
        return get_dist(
            self.verts[self.SHOULDER_BREADTH[0]],
            self.verts[self.SHOULDER_BREADTH[1]]
        )

    @property
    def chest_width(self):
        return get_dist(
            self.verts[self.CHEST_WIDTH[0]],
            self.verts[self.CHEST_WIDTH[1]]
        )

    @property
    def waist_width(self):
        return get_dist(
            self.verts[self.WAIST_WIDTH[0]],
            self.verts[self.WAIST_WIDTH[1]]
        )

    @property
    def torso_depth(self):
        return get_dist(
            self.verts[self.TORSO_DEPTH[0]],
            self.verts[self.TORSO_DEPTH[1]]
        )

    @property
    def hip_width(self):
        return get_dist(
            self.verts[self.HIP_WIDTH[0]],
            self.verts[self.HIP_WIDTH[1]]
        )

    @property
    def arm_span_fingers(self):
        return get_dist(
            self.verts[self.ARM_SPAN_FINGERS[0]],
            self.verts[self.ARM_SPAN_FINGERS[1]]
        )

    @property
    def arm_span_wrist(self):
        return get_dist(
            self.verts[self.ARM_SPAN_WRIST[0]],
            self.verts[self.ARM_SPAN_WRIST[1]]
        )

    @property
    def arm_length(self):
        return get_dist(
            self.verts[self.ARM_LENGTH[0]],
            self.verts[self.ARM_LENGTH[1]]
        )

    @property
    def forearm_length(self):
        return get_dist(
            self.verts[self.FOREARM_LENGTH[0]],
            self.verts[self.FOREARM_LENGTH[1]]
        )

    @property
    def inside_leg_length(self):
        return get_height(
            self.verts[self.INSIDE_LEG_LENGTH[0]],
            self.verts[self.INSIDE_LEG_LENGTH[1]]
        )

    @property
    def waist_circumference(self):
        line_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.LOWER_BELLY_POINT])
        return sum([get_dist(x[0], x[1]) for x in line_segments])

    @property
    def head_circumference(self):
        line_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.FOREHEAD_POINT])
        return sum([get_dist(x[0], x[1]) for x in line_segments])

    @property
    def neck_circumference(self):
        line_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.NECK_POINT])
        return sum([get_dist(x[0], x[1]) for x in line_segments])

    @property
    def chest_circumference(self):
        line_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.LEFT_CHEST])
        return sum([get_dist(x[0], x[1]) for x in line_segments])

    @property
    def pelvis_circumference(self):
        line_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.PELVIS_POINT])
        return sum([get_dist(x[0], x[1]) for x in line_segments])

    @property
    def wrist_circumference(self):
        line_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            VERT_NORMAL,
            self.verts[self.LEFT_WRIST])
        return sum([get_dist(x[0], x[1]) for x in line_segments])

    @property
    def bicep_circumference(self):
        line_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            VERT_NORMAL,
            self.verts[self.RIGHT_BICEP_POINT])
        return sum([get_dist(x[0], x[1]) for x in line_segments])

    @property
    def forearm_circumference(self):
        line_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            VERT_NORMAL,
            self.verts[self.RIGHT_FOREARM_POINT])
        return sum([get_dist(x[0], x[1]) for x in line_segments])

    @property
    def thigh_circumference(self):
        line_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.RIGHT_THIGH_POINT])
        return sum([get_dist(x[0], x[1]) for x in line_segments]) / 2.

    @property
    def calf_circumference(self):
        line_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.RIGHT_CALF_POINT])
        return sum([get_dist(x[0], x[1]) for x in line_segments]) / 2.

    @property
    def ankle_circumference(self):
        line_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.RIGHT_ANKLE_POINT])
        return sum([get_dist(x[0], x[1]) for x in line_segments]) / 2.

    @property
    def allmeasurements(self):
        return np.array([getattr(self, x) for x in dir(self) if '_' in x and x[0].islower()])

    @property
    def apmeasurements(self):
        return np.array([getattr(self, x) for x in MeshMeasurements.aplabels()])

    @staticmethod
    def alllabels():
        return [x for x in dir(MeshMeasurements) if '_' in x and x[0].islower()]

    @staticmethod
    def aplabels():
        return [
            'head_circumference',
            'neck_circumference',
            'shoulder_to_crotch',
            'chest_circumference',
            'waist_circumference',
            'pelvis_circumference',
            'wrist_circumference',
            'bicep_circumference',
            'forearm_circumference',
            'arm_length',
            'inside_leg_length',
            'thigh_circumference',
            'calf_circumference',
            'ankle_circumference',
            'shoulder_breadth'
        ]


def load(args):
    # TODO: Use both data, not only male data.
    data_dir = os.path.join(args.data_root, args.dataset_name, 'prepared', 'male')
    regressor_path = os.path.join(data_dir, 'weights.npy')

    data_dict = {}
    for fname in os.listdir(data_dir):
        data_dict[fname.split('.')[0]] = np.load(os.path.join(data_dir, fname))

    if not os.path.exists(regressor_path):
        weights_in = []
        measurements_all = []

        for sample_idx in range(data_dict['genders'].shape[0]):
            verts = data_dict['vertss'][sample_idx]
            faces = data_dict['facess'][sample_idx]
            gender = data_dict['genders'][sample_idx]

            mesh_measurements = MeshMeasurements(gender, verts, faces, noise_std=2.)

            weights_in.append(mesh_measurements.weight)
            measurements_all.append(mesh_measurements.apmeasurements)

        weights_in = np.array(weights_in).reshape((-1, 1))
        measurements_all = np.array(measurements_all)

        np.save(os.path.join(data_dir, 'weights.npy'), weights_in)
        np.save(os.path.join(data_dir, 'measurements.npy'), measurements_all)
    else:
        weights_in = np.load(regressor_path)
        measurements_all = np.load(os.path.join(data_dir, 'measurements.npy'))

    shapes = data_dict['shapes'][:, 0] if len(data_dict['shapes'].shape) == 3 else data_dict['shapes']
    return weights_in, shapes, measurements_all, data_dict['genders']


def load_star(args):

    def prepare_in(verts, faces, volume, gender, args):
        mesh_measurements = MeshMeasurements(verts, faces, volume)

        weight = (1000 * mesh_measurements.weight) + np.random.normal(0, 0.5)

        return regressor.get_data(), mesh_measurements.allmeasurements

    # TODO: Use both data, not only male data.
    data_dir = os.path.join(args.data_root, args.dataset_name, 'prepared', 'male')

    regressor_name = f'{args.pose_reg_type}_{args.silh_reg_type}_{args.soft_reg_type}.npy'
    regressor_path = os.path.join(data_dir, regressor_name)

    data_dict = {}
    for fname in os.listdir(data_dir):
        data_dict[fname.split('.')[0]] = np.load(os.path.join(data_dir, fname))

    if not os.path.exists(regressor_path):
        samples_in = []
        measurements_all = []

        for sample_idx in range(data_dict['vertss'].shape[0]):
            verts = data_dict['vertss'][sample_idx]
            faces = data_dict['facess'][sample_idx]
            #volume = data_dict['volumes'][sample_idx]
            #gender = data_dict['genders'][sample_idx]
            gender = 'male'

            sample_in, sample_measurements = prepare_in(verts, faces, gender, args)

            samples_in.append(sample_in)
            measurements_all.append(sample_measurements)

        samples_in = np.array(samples_in)
        measurements_all = np.array(measurements_all)

        np.save(regressor_path, samples_in)
        np.save(os.path.join(data_dir, 'measurements.npy'), measurements_all)
    else:
        samples_in = np.load(regressor_path)
        measurements_all = np.load(os.path.join(data_dir, 'measurements.npy'))

    samples_in = normalize(samples_in, axis=0)

    # NOTE: Measurements are here in case I want to experiment with regressing to them instead of shape coefs.
    #return samples_in, data_dict['shapes'][:, 0], measurements_all, np.array([0] * samples_in.shape[0])
    return samples_in, data_dict['shapes'], measurements_all, np.array([0] * samples_in.shape[0])

