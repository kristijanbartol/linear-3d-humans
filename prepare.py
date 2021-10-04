import os
import json
import numpy as np
from pyrender.mesh import Mesh


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


# Mesh measurement indexes.
class MeshMeasurements:

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

    def __init__(self):
        self.overall_height = None
        self.nipple_height = None
        self.navel_height = None
        self.inseam_height = None

        self.shoulder_width = None
        self.chest_width = None
        self.waist_width = None
        self.torso_depth = None
        self.hip_width = None

        self.arm_span_fingers = None
        self.arm_span_wrist = None
        self.arm_length = None
        self.forearm_length = None

        self.volume = None


def extract_measurements(vertices, volume):
    # 1. Overall height
    # 2. Nipple height
    # 3. Navel height
    # 4. Inseam height
    # 5. Shoulder width
    # 6. Chest width (at nipple height)
    # 7. Waist width (at navel height)
    # 8. Torso depth (at navel height)
    # 9. Hip width (at inseam height)
    # 10. Arm span fingers (finger to finger)
    # 11. Arm span wrist (wrist to wrist)
    # 12. Arm length (shoulder to wrist)
    # 13. Forearm length (inner elbow to wrist)
    # 14. Volume
    measurements = MeshMeasurements()

    for static_attr in [x for x in dir(MeshMeasurements) if not x.startswith('__')]:
        indexes = getattr(MeshMeasurements, static_attr)
        distance = np.linalg.norm(vertices[indexes[0]] - vertices[indexes[1]])
        setattr(measurements, static_attr.lower(), distance)

    measurements.volume = volume
    return measurements


def save_measurements(save_dir, measurements: MeshMeasurements):
    attr_list = [x for x in dir(measurements) if not x.startswith('__') and x[0].islower()]
    measure_dict = dict(zip(attr_list, [str(getattr(measurements, x)) for x in attr_list]))
    with open(os.path.join(save_dir, 'measurements.json'), 'w') as json_file:
        json.dump(measure_dict, json_file)


if __name__ == '__main__':
    measurements = MeshMeasurements()
    print('')
