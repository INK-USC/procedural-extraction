import pickle
import numpy as np

dataset = {
    'next': [],
    'if': [],
    'none': []
}

for id in [1,2]:
    with open('dataset/relation/%02d.part.relation.pkl' % id, 'rb') as f:
        ds = pickle.load(f)
    for (k, v) in ds.items():
        dataset[k].extend(v)

for (k, v) in dataset.items():
    np.random.shuffle(v)
    if k == 'none':
        v = v[:int(len(v)/300)]
    if k == 'if':
        continue
    len_part = int(len(v)/8)
    with open('dataset/relation/train.tsv', 'a') as f:
        for triplet in v[:-2 * len_part]:
            f.write("%s\t%s\t%s\n" % triplet)
    with open('dataset/relation/dev.tsv', 'a') as f:
        for triplet in v[-2 * len_part : - len_part]:
            f.write("%s\t%s\t%s\n" % triplet)
    with open('dataset/relation/test.tsv', 'a') as f:
        for triplet in v[- len_part : ]:
            f.write("%s\t%s\t%s\n" % triplet)