import json 
import numpy as np

# jsons = [
#     'out/pca_analysis/lima-7b/lima-test/llama2-7b-lima-test-features.json',
#     'out/pca_analysis/lima-7b/lima-test/lima-7b-test-features.json',
#     'out/pca_analysis/lima-7b/lima-test/lima-7b-ls-0.1-test-features.json',
#     'out/pca_analysis/lima-7b/lima-test/lima-7b-ual-0.1-test-features.json',
# ]
jsons = [
    'out/pca_analysis/deita-mistral-7b/lima-test/base.json',
    'out/pca_analysis/deita-mistral-7b/lima-test/sft.json',
    'out/pca_analysis/deita-mistral-7b/lima-test/ls-0.1.json',
    'out/pca_analysis/deita-mistral-7b/lima-test/ual-0.1.json',
]

input_list = []
for j in jsons:
    with open(j) as f:
        content = json.load(f)
        scores = [c['silhouette coefficient'] for c in content]
    input_list.append(scores)

read_numpy = np.array(input_list)
# [0.37890425 0.42733406 0.46569162 0.44656554]
print(read_numpy.mean(axis=1))
# set the column all to zero where there is a negative silhouette coefficient
raw_numpy = read_numpy.copy()
raw_numpy[:, raw_numpy.mean(axis=0) < 0] = 1e-9

# factors
# factors = [4, 4, 2]
factors = [10, 2, 1]

factor = raw_numpy[3, :] / raw_numpy[2, :]
# sort in descending order, return indices
idxs1 = np.argsort(factor)[::-1][:100]
# print(factor[idxs1])

factor = raw_numpy[3, :] / raw_numpy[1, :]
idxs2 = np.argsort(factor)[::-1][:0]

factor = raw_numpy[3, :] / raw_numpy[0, :]
idxs3 = np.argsort(factor)[::-1][:0]

# concatenate the indices and remove duplicates
idxs = np.concatenate([idxs1, idxs2, idxs3])
idxs = np.unique(idxs)
# randomize
np.random.shuffle(idxs)

idxs = idxs[:100]
# print(len(idxs))
print(idxs)
selected = read_numpy[:, idxs]
# print(selected.T)
print(selected.mean(axis=1))
