from statistics import median


def hog(real_counts, ideal_probs):
    total_real = 0
    for key in real_counts:
        total_real += real_counts[key]

    med = median(ideal_probs.values())

    hog = 0
    for key in real_counts:
        if key in ideal_probs:
            if ideal_probs[key] > med:
                hog += real_counts[key]
    hog /= total_real
    return hog
