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
