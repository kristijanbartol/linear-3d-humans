import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy
from scipy.spatial.transform import Rotation as rot
import torch
import h5py
import json
from random import random

from const import PELVIS, H36M_KPTS_15, H36M_PARTS_15, KPTS_17, BODY_PARTS_17, \
        OPENPOSE_PARTS_15, RADIUS, K
#from data_utils import generate_uniform_projection_matrices, project, \
#        normalize_3d_numpy, generate_uniform_projections_torch


PEOPLE3D_H = 480
PEOPLE3D_W = 640
H36M_H = 1000
H36M_W = 1002


def to_int_tuple(xs):
    return tuple([int(x) for x in xs])


def draw_2d(kpts_2d, kpts_set=KPTS_17, parts_set=BODY_PARTS_17, prev_img=None):
    if prev_img is None:
        img = np.zeros((H, W, 3), np.uint8)
    else:
        img = deepcopy(prev_img)

    for kpt_idx, kpt in enumerate(kpts_2d):
        kpt = kpt[:2]
        if kpt_idx in kpts_set:
            if kpt_idx == PELVIS - 1:
                img = cv2.circle(
                        img, to_int_tuple(kpt),
                        radius=1, color=(0, 0, 255), thickness=-1)
            else:
                img = cv2.circle(
                    img, tuple([int(x) for x in kpt]),
                    radius=1, color=(0, 255, 0), thickness=-1)

    for part in parts_set:
        start_point = to_int_tuple(kpts_2d[part[0]][:2])
        end_point = to_int_tuple(kpts_2d[part[1]][:2])
        img = cv2.line(img, start_point, end_point, (255, 0, 0), thickness=1) 

    if prev_img is not None:
        return img
    else:
        cv2.imshow('2d keypoints', img)


def draw_3d(kpts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for kpt in kpts:
        ax.scatter(kpt[0], kpt[1], kpt[2], c='r', marker='o')

    for part in BODY_PARTS_17:
        start_point = to_int_tuple(kpts_2d[part[0]-1])
        end_point = to_int_tuple(kpts_2d[part[1]-1])
        img = cv2.line(img, start_point, end_point, (255, 0, 0), thickness=1) 

    #plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('square')
    ax.set_title('3D vis')

    plt.show()


def read_txt(fpath):
    kpts_2d = []
    kpts_3d = []
    with open(fpath) as pose_f:
        lines = pose_f.readlines()[1:]

        for line in lines:
            kpts_2d.append([float(c) for c in line[:-1].split(' ')][:2])
            kpts_3d.append([float(c) for c in line[:-1].split(' ')][3:])
    return kpts_2d, kpts_3d


def read_numpy(fpath):
    kpts_3d = np.load(fpath)[0].reshape((17, 3))
    return kpts_3d


def draw_numpy_2d(kpts_2d, h, w):
    img = np.zeros((h, w, 3), np.uint8)

    for idx in range(kpts_2d.shape[0]):
        coordinates = tuple([int(x) for x in kpts_2d[idx]][:2])
        if idx + 1 in KPTS_17:
            img = cv2.circle(
                img, coordinates,
                radius=1, color=(0, 255, 0), thickness=-1)

    for part in BODY_PARTS_17:
        start_point = to_int_tuple(kpts_2d[part[0]-1][:2])
        end_point = to_int_tuple(kpts_2d[part[1]-1][:2])
        img = cv2.line(img, start_point, end_point, (255, 0, 0), thickness=1) 

    cv2.imshow('2d keypoints', img)


def draw_openpose(json_fpath, img_path):
    # TODO: Put original image as background.
    #img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    img = cv2.imread(img_path)

    with open(json_fpath) as fjson:
        data = json.load(fjson)
    pose_2d = np.array(data['people'][0]['pose_keypoints_2d'], dtype=np.float32)
    pose_2d = np.delete(pose_2d, np.arange(2, pose_2d.size, 3))
    print(pose_2d)
    for idx in range(int(pose_2d.shape[0] / 2)):
        coord = (pose_2d[idx*2], pose_2d[idx*2+1])
        print(coord)
        img = cv2.circle(img, coord, radius=1, color=(0, 255, 0), thickness=-1)

    for part in OPENPOSE_PARTS_15:
        start_point = (pose_2d[part[0]*2], pose_2d[part[0]*2+1])
        end_point = (pose_2d[part[1]*2], pose_2d[part[1]*2+1])
        img = cv2.line(img, start_point, end_point, (255, 0, 0), thickness=1)

    cv2.imshow('2d keypoints', img)
    cv2.waitKey(0)


def draw_keypoints(pose_2d, render_path):
    img = cv2.imread(render_path)
    for kpt_idx in SMPL_KPTS:
        img = cv2.circle(img, pose_2d[kpt_idx], radius=1, color=(0, 255, 0), thickness=-1)

    for part in SMPL_PARTS:
        img = cv2.line(img, pose_2d[part[0]], pose_2d[part[1]], (255, 0, 0), thickness=1)

    cv2.imshow('2d keypoints', img)
    cv2.waitKey(0)
        


if __name__ == '__main__':
    for subject in ['female0000', 'female0001', 'female0003',
            'male0000', 'male0002', 'male0004']:
        for pose in [0, 2, 3, 4, 6, 9]:
            kpts_3d = np.load(
                    f'data/test/gt/{subject}/0000000{pose}.npy')
            render_path = f'data/test/imgs/{subject}/0/0000000{pose}.png'

            P = generate_uniform_projection_matrices(1)
            kpts_2d = project(kpts_3d, P)
            draw_keypoints(kpts_2d, render_path)

