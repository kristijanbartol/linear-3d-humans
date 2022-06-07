import numpy as np
import itertools
from sympy.utilities.iterables import multiset_permutations


def all_combinations_with_permutations(values, num_coefs):
    all_combinations = []
    for comb in list(itertools.combinations_with_replacement(values, num_coefs)):
        all_combinations += multiset_permutations(comb)
    return all_combinations


def img_to_silhouette(img):
    img = img.copy()
    img.setflags(write=1)
    img[img == 255] = 0
    img[img > 0] = 1
    
    return img[:, :, 0]


def get_dist(vs):
    # NOTE: Works both for 3D and 2D joint coordinates.
    total_dist = 0
    for vidx in range(len(vs) - 1):
        total_dist += np.linalg.norm(vs[vidx] - vs[vidx + 1])
    return total_dist


def get_dist_parallel(vs):
    vs_batched = np.expand_dims(vs, 0)
    return np.sum(np.linalg.norm(vs_batched[:, :-1] - vs_batched[:, 1:], axis=2))


def get_segment_length(segments):
    total_dist = 0
    for sidx in range(len(segments)):
        total_dist += np.linalg.norm(segments[sidx][0] - segments[sidx][1])
    return total_dist


def get_height(v1, v2):
    # NOTE: Works both for 3D and 2D joint coordinates.
    return np.abs((v1 - v2))[1]
