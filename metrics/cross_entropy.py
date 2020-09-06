from functools import reduce
from math import log


def cross_entropy(real_counts, ideal_probs):
    total_real = reduce(lambda a, b: a + b, real_counts.values())
    ce = reduce(lambda a, b: a+b, list(map(lambda x: real_counts[x]*log(1/ideal_probs[x], 2), real_counts)))
    return ce / total_real
