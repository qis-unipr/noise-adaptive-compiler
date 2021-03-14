def l1_norm(real_counts, ideal_probs):
    total_real = 0
    for key in real_counts:
        total_real += real_counts[key]

    n_bits = len(list(ideal_probs.keys())[0])
    keys = []
    for i in range(2**n_bits):
        keys.append(bin(i).split('b')[1].zfill(n_bits))

    l1n = 0
    for key in keys:
        if key in real_counts and key in ideal_probs:
            l1n += abs(real_counts[key]/total_real - ideal_probs[key])
        elif key in real_counts:
            l1n += abs(real_counts[key]/total_real)
        elif key in ideal_probs:
            l1n += abs(0.0 - ideal_probs[key])
    l1n /= total_real
    return l1n
