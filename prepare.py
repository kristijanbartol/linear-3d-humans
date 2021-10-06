import os
import json
import numpy as np
from pyrender.mesh import Mesh


def get_dist(v1, v2):
    return np.linalg.norm(v1 - v2)


def get_height(v1, v2):
    return np.abs((v1 - v2))[1]


class MeshMeasurements:

    # Mesh landmark indexes.
    HEAD_TOP = 1390
    LEFT_HEEL = 3576
    LEFT_NIPPLE = 1067
    BELLY_BUTTON = 5049
    INSEAM_POINT = 1884
    LEFT_SHOULDER = 2940
    RIGHT_SHOULDER = 4023
    LEFT_CHEST = 7361
    RIGHT_CHEST = 6790
    LEFT_WAIST = 10323
    RIGHT_WAIST = 7051
    UPPER_BELLY_POINT = 7921
    REVERSE_BELLY_POINT = 4722
    LEFT_HIP = 6241
    RIGHT_HIP = 2732
    LEFT_MID_FINGER = 4340
    RIGHT_MID_FINGER = 2924
    LEFT_WRIST = 6243
    RIGHT_WRIST = 9926
    LEFT_INNER_ELBOW = 1887
    RIGHT_INNER_ELBOW = 5082

    # Mesh measurement idnexes.
    OVERALL_HEIGHT = (HEAD_TOP, LEFT_HEEL)
    NIPPLE_HEIGHT = (LEFT_NIPPLE, LEFT_HEEL)
    NAVEL_HEIGHT = (BELLY_BUTTON, LEFT_HEEL)
    INSEAM_HEIGHT = (INSEAM_POINT, LEFT_HEEL)

    SHOULDER_WIDTH = (LEFT_SHOULDER, RIGHT_SHOULDER)
    CHEST_WIDTH = (LEFT_CHEST, RIGHT_CHEST)
    WAIST_WIDTH = (LEFT_WAIST, RIGHT_WAIST)
    TORSO_DEPTH = (UPPER_BELLY_POINT, REVERSE_BELLY_POINT)
    HIP_WIDTH = (LEFT_HIP, RIGHT_HIP)

    ARM_SPAN_FINGERS = (LEFT_MID_FINGER, RIGHT_MID_FINGER)
    ARM_SPAN_WRIST = (LEFT_WRIST, RIGHT_WRIST)
    ARM_LENGTH = (LEFT_SHOULDER, LEFT_WRIST)
    FOREARM_LENGTH = (LEFT_INNER_ELBOW, LEFT_WRIST)

    def __init__(self, verts, volume):
        self.verts = verts
        self.weight = volume

    @property
    def overall_height(self):
        return get_height(
            self.verts[self.OVERALL_HEIGHT[0]], 
            self.verts[self.OVERALL_HEIGHT[1]]
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
    def shoulder_width(self):
        return get_dist(
            self.verts[self.SHOULDER_WIDTH[0]],
            self.verts[self.SHOULDER_WIDTH[1]]
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


class PoseFeatures():

    # Joint indexes.
    HEAD = 15
    LEFT_HEEL = 62
    LEFT_MIDDLE = 68
    RIGHT_MIDDLE = 73
    SPINE3 = 9  # nipple height
    LEFT_HIP = 1
    RIGHT_HIP = 2
    LEFT_WRIST = 20
    LEFT_SHOULDER = 16

    # Joint-based measurement indexes.
    OVERALL_HEIGHT = (HEAD, LEFT_HEEL)
    ARM_SPAN_FINGERS = (LEFT_MIDDLE, RIGHT_MIDDLE)
    INSEAM_HEIGHT = (SPINE3, LEFT_HEEL)
    HIPS_WIDTH = (LEFT_HIP, RIGHT_HIP)
    ARM_LENGTH = (LEFT_WRIST, LEFT_SHOULDER)

    def __init__(self, joints):
        self.joints = joints

    @property
    def overall_height(self):
        return get_height(
            self.joints[self.OVERALL_HEIGHT[0]], 
            self.joints[self.OVERALL_HEIGHT[1]]
        )

    @property
    def arm_span_fingers(self):
        return get_dist(
            self.joints[self.ARM_SPAN_FINGERS[0]], 
            self.joints[self.ARM_SPAN_FINGERS[1]]
        )

    @property
    def inseam_height(self):
        return get_height(
            self.joints[self.INSEAM_HEIGHT[0]], 
            self.joints[self.INSEAM_HEIGHT[1]]
        )

    @property
    def hips_width(self):
        return get_dist(
            self.joints[self.HIPS_WIDTH[0]], 
            self.joints[self.HIPS_WIDTH[1]]
        )

    @property
    def arm_length(self):
        return get_dist(
            self.joints[self.ARM_LENGTH[0]], 
            self.joints[self.ARM_LENGTH[1]]
        )


# TODO: Implement SilhouetteFeatures.
class SilhouetteFeatures():

    def __init__(self, silhouettes):
        pass


class Regressors():

    def __init__(self, pose_features, silhouette_features, weight):
        self.pose_features = pose_features
        self.weight = weight

    @property
    def R2(self):
        # [overall height, weight]
        overall_height = self.pose_features.overall_height
        weight = self.weight
        return np.array([overall_height, weight], dtype=np.float32)

    @property
    def R4(self):
        # [overall height, weight, arm span fingers, inseam height]
        arm_span_fingers = self.pose_features.arm_span_fingers
        inseam_height = self.pose_features.inseam_height
        return np.concatenate(
            [self.R2, np.array([arm_span_fingers, inseam_height], 
            dtype=np.float32)]
        )

    @property
    def R5(self):
        # [overall height, weight, arm span fingers, inseam height,
        # hips width]
        hips_width = self.pose_features.hips_width
        return np.concatenate(
            [self.R4, np.array([hips_width], dtype=np.float32)]
        )

    @property
    def R6(self):
        # [overall height, weight, arm span fingers, inseam height,
        # hips width, arm length]
        arm_length = self.pose_features.arm_length
        return np.concatenate(
            [self.R5, np.array([arm_length], dtype=np.float32)]
        )


# NOTE: For now, use shape parameters as output.
def prepare_in(sample_dict, regressor_type='R4'):
    pose_features = PoseFeatures(sample_dict['joints'])
    silhouette_features = SilhouetteFeatures(sample_dict['silhouettes'])
    measurements = MeshMeasurements(sample_dict['verts'], sample_dict['volume'])
    regressors = Regressors(pose_features, silhouette_features, measurements.weight)
    return getattr(regressors, regressor_type), measurements


def prepare(args):
    data_dir = os.path.join(args.data_root, args.dataset_name, 'gt')
    samples_in = []
    samples_out = []
    measurements_all = []

    for subj_dirname in os.listdir(data_dir):
        subj_dirpath = os.path.join(data_dir, subj_dirname)
        sample_dict = {
            'faces': None,
            'joints': None,
            'pose': None,
            'shape': None,
            'silhouettes': None,
            'verts': None,
            'volume': None  # optional
        }

        for fname in os.listdir(subj_dirpath):
            key = fname.split('.')[0].split('_')[0]
            data = np.load(os.path.join(subj_dirpath, fname))
            sample_dict[key] = data

        sample_in, measurements_object = prepare_in(sample_dict, args.regressor_type)

        sample_measurements = [getattr(measurements_object, x) for x in dir(measurements_object) if '_' in x and x[0].islower()]

        samples_in.append(sample_in)
        measurements_all.append(sample_measurements)
        samples_out.append(sample_dict['shape'])

    # TODO: Move index selection of samples_out to generate.py
    return np.array(samples_in), np.array(samples_out)[:, 0], np.array(measurements_all)
