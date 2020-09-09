from functools import reduce


def l1_norm(real_counts, ideal_probs):
    total_real = reduce(lambda a, b: a + b, real_counts.values())
    n_bits = len(list(ideal_probs.keys())[0])
    keys = []
    for i in range(2**n_bits):
        keys.append(bin(i).split('b')[1].zfill(n_bits))
    real_ideal = {}
    for key in keys:
        if key in real_counts and key in ideal_probs:
            real_ideal[key] = (real_counts[key] / total_real, ideal_probs[key])
        elif key in real_counts:
            real_ideal[key] = (real_counts[key] / total_real, 0.0)
        else:
            real_ideal[key] = (0.0, ideal_probs[key])
    l1n = reduce(lambda a, b: abs(a) + abs(b), list(map(
        lambda x: real_ideal[x][0] - real_ideal[x][1], real_ideal)))
    return l1n
