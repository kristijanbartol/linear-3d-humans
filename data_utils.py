import os
import numpy as np
from copy import deepcopy
import cv2
from scipy.spatial.transform import Rotation as rot
import torch, torchvision
from time import time
import math
from random import random

from const import H, W, PELVIS, RADIUS, K, SMPL_KPTS_15, SMPL_PARTS
from vis import draw_2d


def rot_from_euler(x, y, z):
    cosx = torch.cos(torch.deg2rad(x))
    #cosx = torch.cos(x)
    sinx = torch.sin(torch.deg2rad(x))
    #sinx = torch.sin(x)

    cosy = torch.cos(torch.deg2rad(y))
    #cosy = torch.cos(y)
    siny = torch.sin(torch.deg2rad(y))
    #siny = torch.sin(y)

    cosz = torch.cos(torch.deg2rad(z))
    #cosz = torch.cos(z)
    sinz = torch.cos(torch.deg2rad(z))
    #sinz = torch.cos(z)

    Rx = torch.tensor([
        [1., 0., 0.],
        [0., cosx, -sinx],
        [0., sinx, cosx]
    ])

    Ry = torch.tensor([
        [cosy, 0, siny],
        [0., 1., 0.],
        [-siny, 0., cosy]
    ])

    Rz = torch.tensor([
        [cosz, -sinz, 0.],
        [sinz, cosz, 0.],
        [0., 0., 1.]
    ])

    R = torch.matmul(Rz, Ry)
    R = torch.matmul(R, Rx)

    return R


def rot_from_euler_numpy(x, y, z):
    cosx = np.cos(np.deg2rad(x))
    sinx = np.sin(np.deg2rad(x))

    cosy = np.cos(np.deg2rad(y))
    siny = np.sin(np.deg2rad(y))

    cosz = np.cos(np.deg2rad(z))
    sinz = np.cos(np.deg2rad(z))

    
    Rx = np.array([
        [1., 0., 0.],
        [0., cosx, -sinx],
        [0., sinx, cosx]
    ])

    Ry = np.array([
        [cosy, 0, siny],
        [0., 1., 0.],
        [-siny, 0., cosy]
    ])

    Rz = np.array([
        [cosz, -sinz, 0.],
        [sinz, cosz, 0.],
        [0., 0., 1.]
    ])

    R = np.dot(Rz, np.dot(Ry, Rx))
    return R



def create_projection_matrix(x, y):
    #y = np.sqrt(RADIUS ** 2 - x ** 2 - z ** 2)
    T = np.array([0., 0., -RADIUS]).reshape((3, 1))
    #T = torch.tensor([0., 0., -RADIUS]).reshape((3, 1))
    #R = rot.from_euler('xyz', [-60, 0, 0], degrees=True).as_matrix()
    if abs(y) < 25 and x > -80:
        z = y
    else:
        z = 90 + x if y > 0 else -90 - x
    R = rot.from_euler('xyz', [x, y, z], degrees=True).as_matrix()
    #R = rot_from_euler(x, y, z)
    R = rot_from_euler_numpy(x, y, z)
    
    RT = np.hstack((R, T))
    #RT = torch.cat((R, T), dim=1)

    #return torch.mm(torch.tensor(K), RT)
    return np.dot(np.array(K), RT)


def generate_random_projection():
    # NOTE: Not going too high with x.
    x = np.random.random_sample() * 45 - 90
    #x = torch.from_numpy(np.array(np.random.random_sample() * 45 - 90, dtype=np.float32))
    y = np.random.random_sample() * 180 - 90
    #y = torch.from_numpy(np.array(np.random.random_sample() * 180 - 90, dtype=np.float32))
    P = create_projection_matrix(45, 60)
    return P


def create_look_at_matrix_torch(x, y, z):
    from_ = torch.tensor([x, y, z], dtype=torch.float32)
    to = torch.tensor([0., 0., 0.], dtype=torch.float32)
    tmp = torch.tensor([0., 1., 0.], dtype=torch.float32)
    forward = (from_ - to)
    forward = forward / torch.norm(forward)
    right = torch.cross(tmp, forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)

    R = torch.tensor([
        [right[0], up[0], forward[0]],
        [right[1], up[1], forward[1]],
        [right[2], up[2], forward[2]],
        [0.  ,     y   , -RADIUS   ]
    ])
    return torch.transpose(R, 0, 1)


def generate_uniform_projections_torch(num_views):
    # NOTE: Z-axis is depth, Y-axis is height.
    #PI = torch.tensor(3.14159265359)
    Ps = torch.zeros((num_views, 3, 4))
    PI = 3.14159265359
    step = 2 * PI / num_views
    for idx in range(num_views):
        angle = torch.tensor(idx * step)
        if angle < PI / 2:
            z = -RADIUS / torch.sqrt(torch.tan(angle) ** 2 + 1)
            x = z * torch.tan(angle)
        elif angle < PI:
            angle -= PI / 2
            x = -RADIUS / torch.sqrt(torch.tan(angle) ** 2 + 1)
            z = -x * torch.tan(angle)
        elif angle < 3 * PI / 2:
            angle -= PI
            z = RADIUS / torch.sqrt(torch.tan(angle) ** 2 + 1)
            x = z * torch.tan(angle)
        else:
            angle -= 3 * PI / 2
            x = RADIUS / torch.sqrt(torch.tan(angle) ** 2 + 1)
            z = -x * torch.tan(angle)
            
        RT = create_look_at_matrix_torch(x, 0., z)
        Ps[idx] = torch.mm(torch.tensor(K), RT)
    return Ps


def create_look_at_matrix(x, y, z):
    from_ = np.array([x, y, z], dtype=np.float32)
    to = np.array([0., 0., 0.], dtype=np.float32)
    tmp = np.array([0., 1., 0.], dtype=np.float32)
    forward = (from_ - to)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(tmp, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    R = [
        [right[0], up[0], forward[0]],
        [right[1], up[1], forward[1]],
        [right[2], up[2], forward[2]],
        [0.  ,     y   , -RADIUS   ]
    ]
    return np.transpose(R)


def generate_uniform_projection_matrices(num_views, radius=RADIUS, k=K):
    # NOTE: Z-axis is depth, Y-axis is height.
    step = 2 * np.pi / num_views
    Ps = []
    for idx in range(num_views):
        angle = idx * step
        if angle < np.pi / 2:
            z = -RADIUS / np.sqrt(np.tan(angle) ** 2 + 1)
            x = z * np.tan(angle)
        elif angle < np.pi:
            angle -= np.pi / 2
            x = -RADIUS / np.sqrt(np.tan(angle) ** 2 + 1)
            z = -x * np.tan(angle)
        elif angle < 3 * np.pi / 2:
            angle -= np.pi
            z = RADIUS / np.sqrt(np.tan(angle) ** 2 + 1)
            x = z * np.tan(angle)
        else:
            angle -= 3 * np.pi / 2
            x = RADIUS / np.sqrt(np.tan(angle) ** 2 + 1)
            z = -x * np.tan(angle)
            
        RT = create_look_at_matrix(x, 0., z)
        P = np.dot(np.array(k), RT)
        Ps.append(P)
    return np.array(Ps)


def random_translate(kpts_3d):
    t_x = random() * RADIUS / 2 - RADIUS / 4
    t_z = random() * RADIUS / 2 - RADIUS / 4
    kpts_3d[:, 0] += t_x
    kpts_3d[:, 2] += t_z
    return kpts_3d, t_x, t_z


def project(kpts_3d, proj_mat):
    ones_vector = np.ones((kpts_3d.shape[0], 1), dtype=np.float32)
    kpts_3d = np.hstack((kpts_3d, ones_vector))
    kpts_2d = np.dot(kpts_3d, proj_mat.transpose())
    last_row = kpts_2d[:, 2].reshape((kpts_2d.shape[0]), 1)
    kpts_2d_hom = np.multiply(kpts_2d, 1. / last_row)
    return kpts_2d_hom.transpose(0, 1)


def normalize_3d_numpy(x):
    diff_x = np.abs(np.amax(x[:, 0]) - np.amin(x[:, 0]))
    diff_y = np.abs(np.amax(x[:, 1]) - np.amin(x[:, 1]))
    diff_z = np.abs(np.amax(x[:, 2]) - np.amin(x[:, 2]))
    print(diff_x, diff_y, diff_z)
    max_diff = np.amax(np.array([diff_x, diff_y, diff_z]))

    return x / max_diff, max_diff


if __name__ == '__main__':
    #start_time = time()
    #P = generate_random_projection()
    Ps = generate_uniform_projection_matrices(5)
    
    #print('Generating random projection: {}'.format(time() - start_time))
    #P = create_projection_matrix(-90, 0)
    '''
    with open('dataset/3dpeople/train/woman17/02_04_jump/camera01/0026.txt') as kpt_f:
        lines = [x[:-1] for x in kpt_f.readlines()]
        kpts = [[float(x) for x in y.split(' ')] for y in lines[1:]]
        #kpts_3d = np.array([x[3:] for x in kpts]).swapaxes(0, 1)
        kpts_3d = torch.tensor([x[3:] for x in kpts]).transpose(0, 1)
    #start_time = time()
    '''

    kpts_3d = np.load('data/gender/gt/S0/00000000.npy')
    print(kpts_3d)
    for idx in range(5):
        kpts_2d = project(kpts_3d, Ps[idx])
        img = np.zeros((H, W, 3), np.uint8)
        img = draw_2d(kpts_2d, SMPL_KPTS_15, SMPL_PARTS, img)
        print(kpts_2d)
        #img = draw_2d(kpts, img)
        cv2.imshow('2d keypoints', img)
        cv2.waitKey(0)
    #print('Projecting: {}'.format(time() - start_time))

