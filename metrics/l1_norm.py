from functools import reduce


def l1_norm(real_counts, ideal_probs):
    total_real = reduce(lambda a, b: a + b, real_counts.values())
    l1n = reduce(lambda a, b: abs(a) + abs(b), list(map(
        lambda x: real_counts[x] / total_real - ideal_probs[x] if x in real_counts else - ideal_probs[x], ideal_probs)))
    return l1n
