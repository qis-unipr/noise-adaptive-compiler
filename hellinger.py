import pickle as pkl

with open('fake_melbourne_hellinger_results.pkl', 'rb') as f:
    results = pkl.load(f)

print(results)

