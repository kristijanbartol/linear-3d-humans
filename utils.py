import itertools
from sympy.utilities.iterables import multiset_permutations


def all_combinations_with_permutations(values, num_coefs):
    all_combinations = []
    for comb in list(itertools.combinations_with_replacement(values, num_coefs)):
        all_combinations += multiset_permutations(comb)
    return all_combinations
