from functools import reduce
from math import log, exp


def cross_entropy(real_counts, ideal_probs):
    total_real = reduce(lambda a, b: a + b, real_counts.values())
    n_qubits = len(list(ideal_probs.keys())[0])
    ce = reduce(lambda a, b: a+b, list(map(lambda x: log(1/ideal_probs[x], 2) if x in ideal_probs else log(1/exp(-n_qubits), 2), real_counts)))
    return ce / total_real
