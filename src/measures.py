from functools import cached_property
from typing import List, Union
import numpy as np
import trimesh
import torch

from .generate import create_model, set_shape
    
    
INT_TO_GENDER = {
    1: 'male',
    2: 'female'
}

    
class MeasurementType():
    ''' Measurement type class.
    
        Basically an enumeration class with additional checking methods.
    '''
        
    CIRCUMFERENCE = 'circumference'     # maps to _get_dist
    LENGTH = 'length'                   # maps to _get_dist
    BREADTH = 'breadth'                 # maps to _get_dist
    DEPTH = 'depth'                     # maps to _get_dist
    SPAN = 'span'                       # maps to _get_dist
    DISTANCE = 'dist'
    HEIGHT = 'height'
    DEFAULT = 'default'
    
    @staticmethod
    def is_defined(type: str) -> bool:
        ''' Check whether the provided type is defined.
        
            Checks if the type is one of the listed ones or None.
            Otherwise, non-existing.
            
            Parameters
            ----------
            type: str
                The type to check.
            
            Returns
            -------
            is_defined: bool
        '''
        if type in [x for x in MeasurementType.__dict__ if '__' not in x] \
                or type is None:
            return True
        else:
            return False
        
    @staticmethod
    def is_default(type: Union[str, object]) -> bool:
        ''' Check whether the provided type is a default type.
        
            The default type is either `MeasurementType.DEFAULT` or `None`.
            
            Parameters
            ----------
            type: Union[str, object]
                The type to check.
            
            Returns
            -------
                is_default: bool
        '''
        if type is None or type == MeasurementType.DEFAULT:
            return True
        else:
            return False      
    
    @staticmethod
    def is_equal(type: str, other_type: Union[str, object]) -> bool:
        ''' Allow to compare against None using `==`.
        
            Parameters
            ----------
            other: object, 
                The other object which to compare with.
                
            Returns:
                is_equal: bool
        '''
        if other_type is None:
            return MeasurementType.is_default(type)
        else:
            type == other_type

    
class MeshIndices():
    ''' Define all the indices needed to extract the measurements.
    
        An enumeration class with mesh indices corresponding to 
        SMPL vertices.
    '''
    
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
    FOREHEAD_POINT = 336
    NECK_POINT = 3049
    HIP_POINT = 1806
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
    INSIDE_LEG_HEIGHT = (LOW_LEFT_HIP, LEFT_ANKLE)
    
    # Circumference indices.
    WAIST_CIRCUMFERENCE = [3500, 1336, 917, 916, 919, 918, 665, 662, \
            657, 654, 631, 632, 720, 799, 796, 890, 889, 3124, 3018, \
            3019, 3502, 6473, 6474, 6545, 4376, 4375, 4284, 4285, 4208, \
            4120, 4121, 4142, 4143, 4150, 4151, 4406, 4405, \
            4403, 4402, 4812]

    CHEST_CIRCUMFERENCE = [3076, 2870, 1254, 1255, 1349, 1351, 3033, \
            3030, 3037, 3034, 3039, 611, 2868, 2864, 2866, 1760, 1419, \
            741, 738, 759, 2957, 2907, 1435, 1436, 1437, 1252, 1235, 749, \
            752, 3015, 4238, 4237, 4718, 4735, 4736, 4909, ]

    HIP_CIRCUMFERENCE = [1806, 1805, 1804, 1803, 1802, 1801, 1800, 1798, \
            1797, 1796, 1794, 1791, 1788, 1787, 3101, 3114, 3121, 3098, \
            3099, 3159, 6522, 6523, 6542, 6537, 6525, 5252, 5251, 5255, \
            5256, 5258, 5260, 5261, 5264, 5263, 5266, 5265, 5268, 5267]

    THIGH_CIRCUMFERENCE = [877, 874, 873, 848, 849, 902, 851, 852, 897, \
            900, 933, 936, 1359, 963, 908, 911, 1366]

    CALF_CIRCUMFERENCE = [1154, 1372, 1074, 1077, 1470, 1094, 1095, 1473, \
            1465, 1466, 1108, 1111, 1530, 1089, 1086]

    ANKLE_CIRCUMFERENCE = [3322, 3323, 3190, 3188, 3185, 3206, 3182, \
            3183, 3194, 3195, 3196, 3176, 3177, 3193, 3319]

    WRIST_CIRCUMFERENCE = [1922, 1970, 1969, 2244, 1945, 1943, 1979, \
            1938, 1935, 2286, 2243, 2242, 1930, 1927, 1926, 1924]

    FOREARM_CIRCUMFERENCE = [1573, 1915, 1914, 1577, 1576, 1912, 1911, \
            1624, 1625, 1917, 1611, 1610, 1607, 1608, 1916, 1574]

    BICEP_CIRCUMFERENCE = [789, 1311, 1315, 1379, 1378, 1394, 1393, \
            1389, 1388, 1233, 1232, 1385, 1381, 1382, 1397, 1396, 628, 627]

    NECK_CIRCUMFERENCE = [3068, 1331, 215, 216, 440, 441, 452, 218, 219, \
            222, 425, 426, 453, 829, 3944, 3921, 3920, 3734, 3731, 3730, \
            3943, 3935, 3934, 3728, 3729, 4807]

    HEAD_CIRCUMFERENCE = [336, 232, 235, 1, 0, 3, 7, 136, 160, 161, 166, \
            167, 269, 179, 182, 252, 253, 384, 3765, 3766, 3694, 3693, \
            3782, 3681, 3678, 3671, 3672, 3648, 3518, 3513, 3514, 3515, \
            3745, 3744]


class MeshMeasurements():
    ''' The class representing mesh measurements.
    
        Contains all the methods needed to extract body measurements from
        SMPL mesh.
    '''
    
    _LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
               'L', 'M', 'N', 'O', 'P']
    
    _NAMES = [
        'head', 'neck', 'shoulder_to_crotch', 'chest', 'waist', 'hip', 
        'wrist', 'bicep', 'forearm', 'arm', 'inside_leg', 'thigh', 'calf',
        'ankle', 'shoulder', 'overall'
    ]
    
    _LABELS_TO_NAMES = {
        'A': 'head',
        'B': 'neck',
        'C': 'shoulder_to_crotch',
        'D': 'chest',
        'E': 'waist',
        'F': 'hip',
        'G': 'wrist',
        'H': 'bicep',
        'I': 'forearm',
        'J': 'arm',
        'K': 'inside_leg',
        'L': 'thigh',
        'M': 'calf',
        'N': 'ankle',
        'O': 'shoulder',
        'P': 'overall'
    }
    
    _NAMES_TO_LABELS = {
        'head': 'A',
        'neck': 'B',
        'shoulder_to_crotch': 'C',
        'chest': 'D',
        'waist': 'E',
        'hip': 'F',
        'wrist': 'G',
        'bicep': 'H',
        'forearm': 'I',
        'arm': 'J',
        'inside_leg': 'K',
        'thigh': 'L',
        'calf': 'M',
        'ankle': 'N',
        'shoulder': 'O',
        'overall': 'P'
    }  
    
    _DEFAULT_TYPES = {
        'A': MeasurementType.CIRCUMFERENCE,
        'B': MeasurementType.CIRCUMFERENCE,
        'C': MeasurementType.HEIGHT,
        'D': MeasurementType.CIRCUMFERENCE,
        'E': MeasurementType.CIRCUMFERENCE,
        'F': MeasurementType.CIRCUMFERENCE,
        'G': MeasurementType.CIRCUMFERENCE,
        'H': MeasurementType.CIRCUMFERENCE,
        'I': MeasurementType.CIRCUMFERENCE,
        'J': MeasurementType.LENGTH,
        'K': MeasurementType.HEIGHT,
        'L': MeasurementType.CIRCUMFERENCE,
        'M': MeasurementType.CIRCUMFERENCE,
        'N': MeasurementType.CIRCUMFERENCE,
        'O': MeasurementType.BREADTH,
        'P': MeasurementType.HEIGHT
    }
    
    # Measurement type mapping used to call distance calculation function
    # for breadths, depths, and lengths.
    _MEASUREMENT_TYPE_MAPPING = {
        MeasurementType.BREADTH: MeasurementType.DISTANCE,
        MeasurementType.DEPTH: MeasurementType.DISTANCE,
        MeasurementType.LENGTH: MeasurementType.DISTANCE,
        MeasurementType.DISTANCE: MeasurementType.DISTANCE,
        MeasurementType.HEIGHT: MeasurementType.HEIGHT,
        MeasurementType.CIRCUMFERENCE: MeasurementType.DISTANCE
    }
    
    _AVG_HEIGHT = {
        'male': 1.8,
        'female': 1.7
    }
    
    def __init__(
        self,
        gender: str,
        verts: np.ndarray,
        faces: np.ndarray,
        auto_flush: bool = True
    ) -> None:
        ''' Mesh measurements constructor.
        
            Based on given mesh parameters, creates a SMPL model and
            its corresponding vertices and faces. Initializes private
            objects needed for caching and flushing.
            
            Parameters
            ----------
            shape: torch.tensor (Bx10)
                Beta parameters defining body shape.
            gender: str
                Gender of the body model.
            auto_flush: bool
                Whether to flush potentially large data automatically 
                (vertices, faces) or leave the responsibility to the user.
        '''
        self.gender = gender
        self.verts = verts
        self.faces = faces
        self.auto_flush = auto_flush
        self.flushed = False
        
        self._all_measures = {}
        self._height = None
        self._volume = None
        
    @classmethod
    def from_shape_params(
        cls, 
        gender: str,
        shape: torch.tensor, 
        auto_flush: bool = True
    ):
        model = create_model(gender)    
        model_output = set_shape(model, shape)
        
        verts = model_output.vertices.detach().cpu().numpy().squeeze()
        faces = model.faces.squeeze()
        
        return cls(
            gender=gender,
            verts=verts,
            faces=faces,
            auto_flush=auto_flush
        )
        
    @classmethod
    def from_data(
        cls,
        gender: str,
        verts: np.ndarray,
        faces: np.ndarray,
        auto_flush: bool = True
    ):
        return cls(
            gender=gender,
            verts=verts,
            faces=faces,
            auto_flush=auto_flush
        )
        
    def scale_mesh(self, mesh_size):
        '''Scales mesh vertices. Note that all the measurements are updated.'''
        self.verts *= (mesh_size / self.height)
        
    @property
    def height(self):
        ''' Height property of the subject.
        
            Height is the absolute distance between head and heel, i.e.
            the difference between Y values between head and heel
            assuming that the person is upright (and in T-pose).
            
            Returns
            -------
            height: float
                Height value in meters. Note that height is the only
                height-unnormalized body measurement (otherwise the
                height would always be 1. and other measurements
                would remain unnormalized).
        '''
        if self._height is None:
            self._height = self.get_body_measure(
                name_or_label='overall', 
                type='height', 
                normalize=False)
        return self._height
    
    @property
    def volume(self):
        ''' Volume of the body mesh.
            
            Returns
            -------
            volume: float
                The volume is a separate body measurement that corresponds
                to body weight, assuming that the density of the population
                is fixed.
        '''
        if self._volume is None:
            mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)
            self._volume = mesh.volume
        return self._volume
    
    @property
    def weight(self):
        ''' Weight of the subject "approximated" by volume.
            
            Returns
            -------
            volume: float
                The volume is a separate body measurement that corresponds
                to body weight, assuming that the density of the population
                is fixed.
        '''
        return self.volume
        
    def get_body_measure(
            self, 
            name_or_label: str, 
            type: MeasurementType = None,
            normalize: bool = True
        ) -> float:
        ''' Measure body by name/label and measurement type.
        
            Note that this function is not cached because it's not
            that expensive to calculate a single body measurement.
            In case the object is already flushed
        
            Parameters
            ----------
            name_or_label: str, mandatory
                The name (waist, hip, arm, ...) or label (A, B, C, ...)
                of the desired body measurement.
            type: str, optional
                The type of the body measurement (length, breadth, depth,
                or circumference). By default, it returns the most intuitive
                one (see `self._DEFAULT_TYPES`).
                
            Returns
            -------
            body_measurement: float
                Normalized measurement in cm (wrt to average height for 
                particular gender).
        '''
        if len(name_or_label) > 1:
            label = self._NAMES_TO_LABELS[name_or_label]
        elif len(name_or_label) == 1:
            label = name_or_label.upper()
        else:
            raise ValueError('Invalid name/label (empty string?)')
        assert(ord(label) >= ord('A') and ord(label) <= ord('P'))
        
        name = self._LABELS_TO_NAMES[label]
        type = self._verify_type(type, label)
        value_idx = self.all_labels().index(label)
        
        if not MeasurementType.is_default(type) and self.flushed:
            print('WARNING: The object already flushed, returning default-type value.')
            value_idx = self.all_labels().index(label)
            return self._all_measures[None][value_idx]
        
        value = self._total_measure(name, type)
        value = self._norm_measure(value) if normalize else value
        return value
    
    @property
    def all(self, type: str = None) -> List[float]:
        ''' Get all body measurements (their common type is optional).
        
            Note that `self.verts` and `self.faces` are relatively large.
            Therefore, their values are deleted once `self.get_all_measures`
            is called for default type. The default values stay cached and
            they remain safe to fetch.
        
            Parameters
            ----------
            type: str, optional
                The type of the body measurement (length, breadth, depth,
                or circumference). By default, it returns the most intuitive
                one (see `self._DEFAULT_TYPES`).
                
            Returns
            -------
            body_measurement: List[float]
                Normalized body measurements in cm (wrt to the average 
                height for particular gender).
        '''
        type = self._verify_type(type, None)
        
        if not MeasurementType.is_default(type) and self.flushed:
            print('WARNING: The object already flushed, returning default types.')
            return self._all_measures[None]
        
        if type not in self._all_measures:
            values = [self.get_body_measure(x, type) for x in self.all_names()]
            self._all_measures[type] = values

        if self.auto_flush and type is None and not self.flushed:
            self.volume     # Trigger volume calculation in case it wasn't called.
            self.verts = None
            self.faces = None
            self.flushed = True
            
        return self._all_measures[type]
        
    def _total_measure(
            self, 
            measure_name: str, 
            type: str
        ) -> float:
        ''' Measure given body measurement, wrt its type.
        
            Parameters
            ----------
            measure_name: str, mandatory
                The name (waist, hip, arm, ...) of the desired body measurement.
            type: str, optional
                The type of the body measurement (length, breadth, depth,
                or circumference). By default, it returns the most intuitive
                one (see `self._DEFAULT_TYPES`).
                
            Returns
            -------
            body_measurement: float
                Body measure in cm.
        '''
        idx_attr_name = f'{measure_name.upper()}_{type.upper()}'
        _indexes = getattr(MeshIndices, idx_attr_name)
        
        dist_fun_type = self._MEASUREMENT_TYPE_MAPPING[type]
        _dist_fun = getattr(self, f'_get_{dist_fun_type}')
        
        line_segments = np.array([
            (self.verts[_indexes[idx]], self.verts[_indexes[idx+1]]) \
                for idx in range(len(_indexes) - 1)])
        return sum([_dist_fun([x[0], x[1]]) for x in line_segments])
    
    def _norm_measure(self, value: float) -> float:
        ''' Height-normalize the body measurement value.
        
            Only makes sense to apply to measurements other than height. :)
            
            Parameters
            ----------
            value: float
                The given body measurement in meters.
                
            Returns
            -------
            norm_value: float
                Height-normalized body measurement.
        '''
        return value / self.height * self._AVG_HEIGHT[self.gender]
    
    def _verify_type(
            self, 
            type: str, 
            label: str = None
        ) -> str:
        ''' Verify body measurement type value (string).
        
            The type is verified in order to be one of the defined types.
            If the type is default and if the label is not provided, it
            means that all values will be generated by their default
            types, i.e. the method is called from 
            `MeshMeasurements.get_all_measures`.
            
            Parameters
            ----------
            type: str,
                The type that is being verified.
            label: str
                The corresponding label for the particular body measurement.
                Can be `None`, in which case all the measurements will be
                generated with their default types.
            
            Returns
            -------
            verified_type: str
                If the type is incorrect, it simply becomes default type.
                Otherwise, it remains the same.
        '''
        if MeasurementType.is_default(type) and label is None:
            return None
        if MeasurementType.is_defined(type):
            type = self._DEFAULT_TYPES[label]
        return type

    @staticmethod
    def _get_height(vs: List[float]) -> float:
        '''The difference between head and heel (Y-axis, erect pose).'''
        return np.abs((vs[0] - vs[1]))[1]
    
    @staticmethod
    def _get_dist(vs: List[float]) -> float:
        '''The Euclidean distance between two vertices.'''
        return np.linalg.norm(vs[0] - vs[1])
    
    @staticmethod
    def names() -> List[str]:
        '''(Property) Return the list of all the body measurement names.'''
        return list(MeshMeasurements._LABELS_TO_NAMES.values())

    @staticmethod
    def labels() -> List[str]:
        '''Return the list of all the body measurement labels.'''
        return list(MeshMeasurements._NAMES_TO_LABELS.values())
    
    @staticmethod
    def full_names() -> List[str]:
        ''' Return the list of all the full body measurement names.
        
            This means that the name is in the form `f'{name}_{type}`. It
            is convenient for printing (__str__).
        '''
        _all_names = MeshMeasurements._LABELS_TO_NAMES.values()
        corresponding_types = MeshMeasurements._DEFAULT_TYPES.values()
        return [f'{x}_{y}' for (x, y) in zip(_all_names, corresponding_types)]
    
    def __getattr__(self, label: str):
        if ord(label) >= ord('A') and ord(label) <= ord('P'):
            value_idx = self.all_labels().index(label)
            return self._all_measures[None][value_idx]
        else:
            print('WARNING: Unknown attribute.')
            return 0.
    
    def __str__(self) -> str:
        ''' Return string representation when print(measurements) is called.
        
            Returns a multi-line string where each line is in the form
            `f{name}_{default_type}: {value * 100:.2f}cm`.
        '''
        values = self.all
        names = self.names
        _default_types = list(self._DEFAULT_TYPES.values())
        measures_str = ''
        for idx in range(len(values)):
            full_measure_name = f'{names[idx]}_{_default_types[idx]}'
            measures_str += f'{full_measure_name}: {values[idx] * 100.:.2f}cm\n'
        return measures_str


if __name__ == '__main__':
    betas = torch.zeros((1, 10), dtype=torch.float32)
    gender = 'male'
    measurements = MeshMeasurements(shape=betas, gender=gender)
    print(measurements.all)
    print(measurements)
    print(measurements.volume)
    print(measurements.A)
