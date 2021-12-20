import os
import math
import numpy as np
from pyrender.mesh import Mesh
from sklearn.preprocessing import normalize
import trimesh

from generate import GENDER_TO_INT_DICT, create_model, set_shape


def get_dist(*vs):
    # NOTE: Works both for 3D and 2D joint coordinates.
    total_dist = 0
    for vidx in range(len(vs) - 1):
        total_dist += np.linalg.norm(vs[vidx] - vs[vidx + 1])
    return total_dist


def get_height(v1, v2):
    # NOTE: Works both for 3D and 2D joint coordinates.
    return np.abs((v1 - v2))[1]


HORIZ_NORMAL = np.array([0, 1, 0], dtype=np.float32)
VERT_NORMAL = np.array([1, 0, 0], dtype=np.float32)


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

    # For proposer labels.
    overall_height = None

    @staticmethod
    def __init_from_shape__(gender, shape, mesh_size=None, keep_mesh=False):
        model = create_model(gender)
        model_output = set_shape(model, shape)
        verts = model_output.vertices.detach().cpu().numpy().squeeze()
        faces = model.faces.squeeze()

        return MeshMeasurements(verts, faces, mesh_size, keep_mesh)

    def __init__(self, verts, faces, mesh_size=None, keep_mesh=False):
        self.verts = verts
        self.faces = faces

        self.mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)

        self.weight = self.mesh.volume

        self.overall_height = self._get_overall_height()

        if mesh_size is not None:
            self._scale_mesh(mesh_size)

        self.allmeasurements = self._get_all_measurements()
        #self.apmeasurements = self._get_ap_measurements()

        if not keep_mesh:
            self.verts = None
            self.faces = None
            self.mesh = None

    def flush(self):
        self.verts = None
        self.faces = None
        self.mesh = None

    def _scale_mesh(self, mesh_size):
        self.verts *= (mesh_size / self.overall_height)
        self.overall_height = self._get_overall_height()
        self.mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)

    # Use this to obtain overall height, but use overall_height property on the outside.
    def _get_overall_height(self):
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

    def _get_all_measurements(self):
        return np.array([getattr(self, x) for x in dir(self) if '_' in x and x[0].islower()])

    def _get_ap_measurements(self):
        return np.array([getattr(self, x) for x in MeshMeasurements.aplabels()])

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

    @staticmethod
    def letterlabels():
        return [
            'A',
            'B',
            'C',
            'D',
            'E',
            'F',
            'G',
            'H',
            'I',
            'J',
            'K',
            'L',
            'M',
            'N',
            'O'
        ]


class SoftFeatures():

    def __init__(self, gender, weight):
        self.gender = gender
        self.weight = weight


class MeshJointIndexSet():

    HEAD = 15
    LHEEL = 62
    LMIDDLE = 68    # not available in (basic) OpenPose set
    RMIDDLE = 73   # not available in (basic) OpenPose set
    PELVIS = 9          # SPINE3 (nipple height)
    LHIP = 1
    RHIP = 2
    LWRIST = 20
    LSHOULDER = 16

    # Joint-based measurement indexes.
    OVERALL_HEIGHT = [HEAD, LHEEL]
    ARM_SPAN_FINGERS = [LMIDDLE, RMIDDLE]
    INSEAM_HEIGHT = [PELVIS, LHEEL]
    HIPS_WIDTH = [LHIP, RHIP]
    ARM_LENGTH = [LWRIST, LSHOULDER]


class OpenPoseJointIndexSet():

    HEAD = 0    # NOSE
    NECK = 1
    RSHOULDER = 2
    RELBOW = 3
    RWRIST = 4
    LSHOULDER = 5
    LELBOW = 6
    LWRIST = 7
    PELVIS = 8  # MIDHIP
    RHIP = 9
    RKNEE = 10
    RANKLE = 11
    LHIP = 12
    LKNEE = 13
    LANKLE = 14
    REYE = 15
    LEYE = 16
    REAR = 17
    LEAR = 18
    LBIGTOE = 19
    LSMALLTOE = 20
    LHEEL = 21
    RBIGTOE = 22
    RSMALLTOE = 23
    RHEEL = 24
    BACKGROUND = 25

    # Joint-based measurement indexes.
    OVERALL_HEIGHT = [HEAD, LHEEL]
    ARM_SPAN_FINGERS = [LWRIST, LELBOW, LSHOULDER, RSHOULDER, RELBOW, RWRIST]
    INSEAM_HEIGHT = [PELVIS, LHEEL]
    HIPS_WIDTH = [LHIP, RHIP]
    ARM_LENGTH = [LWRIST, LELBOW, LSHOULDER]


class PoseFeatures():

    def __init__(self, joints, index_set, overall_height=None):
        self.joints = joints
        self.index_set = index_set
        self.overall_height_ = overall_height

    @property
    def overall_height(self):
        if self.overall_height_ is not None:
            return self.overall_height_
        else:
            return get_height(
                self.joints[self.index_set.OVERALL_HEIGHT[0]], 
                self.joints[self.index_set.OVERALL_HEIGHT[1]]
            )

    @property
    def arm_span_fingers(self):
        return get_dist(*[self.joints[x] for x in self.index_set.ARM_SPAN_FINGERS])

    @property
    def inseam_height(self):
        return get_height(
            self.joints[self.index_set.INSEAM_HEIGHT[0]], 
            self.joints[self.index_set.INSEAM_HEIGHT[1]]
        )

    @property
    def hips_width(self):
        return get_dist(
            self.joints[self.index_set.HIPS_WIDTH[0]], 
            self.joints[self.index_set.HIPS_WIDTH[1]]
        )

    @property
    def arm_length(self):
        return get_dist(*[self.joints[x] for x in self.index_set.ARM_LENGTH])


class SilhouetteFeatures():

    class __BoundingBox():

        def __init__(self, up, down, left, right):
            self.up = up
            self.down = down
            self.left = left
            self.right = right

    def __init__(self, silhouettes):
        self.silhouettes = silhouettes
        if silhouettes is not None:
            self.bounding_boxes = self.__compute_bounding_boxes()

    def __compute_bounding_boxes(self):
        bounding_boxes = []
        for sidx in range(self.silhouettes.shape[0]):
            up, down, left, right = None, None, None, None
            for row in range(self.silhouettes[sidx].shape[0]):
                if self.silhouettes[sidx][row].sum() != 0:
                    up = row
                    break
            for row in range(self.silhouettes[sidx].shape[0] - 1, 0, -1):
                if self.silhouettes[sidx][row].sum() != 0:
                    down = row
                    break
            for column in range(self.silhouettes[sidx].shape[1]):
                if self.silhouettes[sidx, :, column].sum() != 0:
                    left = column
                    break
            for column in range(self.silhouettes[sidx].shape[1] - 1, 0, -1):
                if self.silhouettes[sidx, :, column].sum() != 0:
                    right = column
                    break
            bounding_boxes.append(self.__BoundingBox(up, down, left, right))
        return bounding_boxes

    @property
    def waist_width(self):
        front_silhouette = self.silhouettes[0]
        bbox = self.bounding_boxes[0]

        row_idx = int(bbox.up + 0.4 * (bbox.down - bbox.up))
        return front_silhouette[row_idx].sum()

    @property
    def waist_depth(self):      # NOTE: Only this is currently using side silhouette!
        side_silhouette = self.silhouettes[1]
        bbox = self.bounding_boxes[1]

        row_idx = int(bbox.up + 0.406 * (bbox.down - bbox.up))
        return side_silhouette[row_idx].sum()

    @property
    def thigh_width(self):
        front_silhouette = self.silhouettes[0]
        bbox = self.bounding_boxes[0]

        row_idx = int(bbox.up + 0.564 * (bbox.down - bbox.up))
        return front_silhouette[row_idx].sum() / 2.

    @property
    def biceps_width(self):
        front_silhouette = self.silhouettes[0]
        bbox = self.bounding_boxes[0]

        column_idx = int(bbox.left + 0.332 * (bbox.right - bbox.left))
        return front_silhouette[:, column_idx].sum()


class Regressor():

    P2 = ['overall_height']
    P4 = P2 + ['arm_span_fingers', 'inseam_height']
    P5 = P4 + ['hips_width']
    P6 = P5 + ['arm_length']

    Si4 = [
        'waist_width',
        'waist_depth',
        'thigh_width',
        'biceps_width'
    ]

    So1 = ['weight']

    def __init__(self, 
            pose_reg_type: str, 
            silh_reg_type: str, 
            soft_reg_type: str,
            pose_features: PoseFeatures, 
            silhouette_features: SilhouetteFeatures, 
            soft_features: SoftFeatures):
        self.pose_reg_type = pose_reg_type
        self.silh_reg_type = silh_reg_type
        self.soft_reg_type = soft_reg_type
        self.pose_features = pose_features
        self.silhouette_features = silhouette_features
        self.soft_features = soft_features

    @property
    def _labels(self):
        pose_labels = getattr(Regressor, self.pose_reg_type) if self.pose_reg_type is not None else []
        silh_labels = getattr(Regressor, self.silh_reg_type) if self.silh_reg_type is not None else []
        soft_labels = getattr(Regressor, self.soft_reg_type) if self.soft_reg_type is not None else []
        return pose_labels, silh_labels, soft_labels

    def get_data(self):
        pose_labels, silh_labels, soft_labels = self._labels
        pose_data = [getattr(self.pose_features, x) for x in pose_labels]
        silh_data = [getattr(self.silhouette_features, x) for x in silh_labels]
        soft_data = [getattr(self.soft_features, x) for x in soft_labels]

        return np.array(pose_data + silh_data + soft_data, dtype=np.float32)

    @staticmethod
    def get_labels(args):
        return ['height', 'weight']


def prepare_in(verts, faces, volume, gender, args):
    mesh_measurements = MeshMeasurements(verts, faces)

    height = mesh_measurements.overall_height + np.random.normal(0, .01)
    weight = (1000 * mesh_measurements.weight) + np.random.normal(0, 1.5)

    return np.array([height, weight, weight / height ** 2, weight * height, weight ** 2, height ** 2, weight ** 2 * height ** 2]), mesh_measurements.apmeasurements


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
            volume = data_dict['volumes'][sample_idx]
            gender = data_dict['genders'][sample_idx]

            sample_in, sample_measurements = prepare_in(verts, faces, volume, gender, args)

            samples_in.append(sample_in)
            measurements_all.append(sample_measurements)

        samples_in = np.array(samples_in)
        measurements_all = np.array(measurements_all)

        np.save(regressor_path, samples_in)
        np.save(os.path.join(data_dir, 'measurements.npy'), measurements_all)
    else:
        samples_in = np.load(regressor_path)
        measurements_all = np.load(os.path.join(data_dir, 'measurements.npy'))

    return samples_in, data_dict['shapes'][:, 0], measurements_all, data_dict['genders']


def prepare_in_from_shapes(args, shape):
    mesh_measurements = MeshMeasurements.__init_from_shape__(args.gender, shape)

    height = mesh_measurements.overall_height + np.random.normal(0, args.height_noise)
    weight = (1000 * mesh_measurements.weight) + np.random.normal(0, args.weight_noise)

    INTERACTION_TERMS = [weight / height ** 2, weight * height, weight ** 2, height ** 2, height / weight]

    return np.array([height, weight] + INTERACTION_TERMS[:args.num_interaction]), mesh_measurements.allmeasurements


def load_from_shapes(args):
    data_dir = os.path.join(args.data_root, args.dataset_name, 'prepared', args.gender)

    suffix = f'_{args.num_interaction}' if args.num_interaction > 0 else ''

    regressor_name = f'inputs_{args.height_noise}_{args.weight_noise}{suffix}.npy'
    regressor_path = os.path.join(data_dir, regressor_name)

    #shapes = np.load(os.path.join(data_dir, 'shapes.npy'))
    shapes = np.load(os.path.join(data_dir, 'shapes.npy'))

    if not os.path.exists(regressor_path):
        samples_in = []
        measurements_all = []

        for shape_idx, shape in enumerate(shapes):
            if shape_idx % 1000 == 0 and shape_idx != 0:
                print(shape_idx)
            sample_in, sample_measurements = prepare_in_from_shapes(args, shape)

            samples_in.append(sample_in)
            measurements_all.append(sample_measurements)

        samples_in = np.array(samples_in)
        measurements_all = np.array(measurements_all)

        np.save(regressor_path, samples_in)
        np.save(os.path.join(data_dir, 'measurements.npy'), measurements_all)
    else:
        samples_in = np.load(regressor_path)
        measurements_all = np.load(os.path.join(data_dir, 'measurements.npy'))

    genders_all = np.array([GENDER_TO_INT_DICT[args.gender]] * samples_in.shape[0])

    # TODO: Update code (generate.py) so that you remove this odd indexing of shapes.
    return samples_in, shapes[:, 0], measurements_all, genders_all
