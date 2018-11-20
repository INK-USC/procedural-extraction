import random
import pickle

chunk_n = 6

sentences = []
for i in range(1, chunk_n+1):
    if i == 5:
        continue
    with open('data/{:02d}.src.txt.dataset.pkl'.format(i), 'rb') as f:
        obj = pickle.load(f)
        sentences.extend(obj)

random.shuffle(sentences)
sen_l = len(sentences)
chunk_l = sen_l // chunk_n

for i in range(0, chunk_n):
    chunk_s = sentences[i * chunk_l: min((i + 1) * chunk_l, sen_l)]
    with open('data/{:02d}.part.dataset.pkl'.format(i+1), 'wb') as f:
        pickle.dump(chunk_s, f)