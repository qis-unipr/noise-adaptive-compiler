from functools import reduce
from statistics import median


def hog(real_counts, ideal_probs):
    total_real = reduce(lambda a, b: a + b, real_counts.values())
    med = median(ideal_probs.values())
    hog = reduce(lambda a,b: a+b, list(map(lambda x: real_counts[x] if real_counts[x]/total_real > med else 0, real_counts)))
    return hog / total_real
