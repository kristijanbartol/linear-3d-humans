# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from typing import NewType, Union, Optional
from dataclasses import dataclass, asdict, fields
import numpy as np
import torch

Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)


@dataclass
class ModelOutput:
    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    transl: Optional[Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


@dataclass
class SMPLOutput(ModelOutput):
    betas: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None


@dataclass
class SMPLHOutput(SMPLOutput):
    left_hand_pose: Optional[Tensor] = None
    right_hand_pose: Optional[Tensor] = None
    transl: Optional[Tensor] = None


@dataclass
class SMPLXOutput(SMPLHOutput):
    expression: Optional[Tensor] = None
    jaw_pose: Optional[Tensor] = None


@dataclass
class MANOOutput(ModelOutput):
    betas: Optional[Tensor] = None
    hand_pose: Optional[Tensor] = None


@dataclass
class FLAMEOutput(ModelOutput):
    betas: Optional[Tensor] = None
    expression: Optional[Tensor] = None
    jaw_pose: Optional[Tensor] = None
    neck_pose: Optional[Tensor] = None


def find_joint_kin_chain(joint_id, kinematic_tree):
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain


def to_tensor(
        array: Union[Array, Tensor], dtype=torch.float32
) -> Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# STAR: Sparse Trained  Articulated Human Body Regressor <https://arxiv.org/pdf/2008.08535.pdf>
#
#
# Code Developed by:
# Ahmed A. A. Osman

import torch

def quat_feat(theta):
    '''
        Computes a normalized quaternion ([0,0,0,0]  when the body is in rest pose)
        given joint angles
    :param theta: A tensor of joints axis angles, batch size x number of joints x 3
    :return:
    '''
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_sin * normalized,v_cos-1], dim=1)
    return quat


def quat2mat(quat):
    '''
        Converts a quaternion to a rotation matrix
    :param quat:
    :return:
    '''
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def rodrigues(theta):
    '''
        Computes the rodrigues representation given joint angles
    :param theta: batch_size x number of joints x 3
    :return: batch_size x number of joints x 3 x 4
    '''
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat2mat(quat)


def with_zeros(input):
    '''
      Appends a row of [0,0,0,1] to a batch size x 3 x 4 Tensor
    :param input: A tensor of dimensions batch size x 3 x 4
    :return: A tensor batch size x 4 x 4 (appended with 0,0,0,1)
    '''
    batch_size  = input.shape[0]
    row_append     = torch.cuda.FloatTensor(([0.0, 0.0, 0.0, 1.0])).to('cpu')
    row_append.requires_grad = False
    padded_tensor     = torch.cat([input, row_append.view(1, 1, 4).repeat(batch_size, 1, 1)], 1)
    return padded_tensor
