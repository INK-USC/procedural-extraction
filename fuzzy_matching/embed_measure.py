"""
Author: Junyi Du
oct. 8 2018
"""

import numpy as np
from tqdm import tqdm
import os.path
import pickle
from typing import List

class EmbeddingMeasurer(object):
    def __init__(self):
        self.memory = {}

    def sen_emb(self, sen: List[str]) -> np.array:
        """
        return average sentence embedding of given sentence
        """
        key = ' '.join(sen)
        if key not in self.memory:
            self.memory[key] = self.L2norm(self.vanilla_sen_emb(sen))
        return self.memory[key]

    def vanilla_sen_emb(self, sen: List[str]) -> np.array:
        raise NotImplementedError("!")
        
    def sim(self, sen1: List[str], sen2: List[str]) -> float:
        """
        return similarity between two sentences
        """
        return self.dot(self.sen_emb(sen1), self.sen_emb(sen2))

    @staticmethod
    def L2norm(embed):
        norm = (embed @ embed)**0.5
        return embed / norm

    @staticmethod
    def dot(self, emb1, emb2):
        """
        return distance between two words
        """
        return emb1 @ emb2


class GloveMeasurer(EmbeddingMeasurer):
    def __init__(self, glove_path='embeddings/glove.840B.300d.txt'):
        super().__init__()

        glove_pkl_path = glove_path + '.pkl'
        if os.path.isfile(glove_pkl_path):
            print('Converted Glove pickle at {} found, loading'.format(glove_pkl_path))
            with open(glove_pkl_path, 'rb') as f:
                self.dictionary = pickle.load(f)
        else:
            print('No converted Glove pickle found')
            self.dictionary = self.load_glove(glove_path)
            print('Saving converted Glove pickle to {}'.format(glove_pkl_path))
            with open(glove_pkl_path, 'wb') as f:
                pickle.dump(self.dictionary, f)
        self.glove_shape = self.dictionary['glove'].shape

        
    def __getitem__(self, item: str) -> np.array:
        """
        entry of all query
        no L2 norm
        """
        word = item.lower()
        if word not in self.dictionary:
            self.dictionary[word] = np.random.normal(size=self.glove_shape)
        word_vec = self.dictionary[word]
        return word_vec

    def vanilla_sen_emb(self, sen: List[str]) -> np.array:
        return np.average(list(map(self.__getitem__, sen)), axis=0)
        
    def load_glove(self, glove_path):
        print('Reading Glove word vectors from {}'.format(glove_path))
        with open(glove_path, 'r') as f:
            lst = f.readlines()

        dictionary = {}

        print('Converting Glove word vectors')
        for line in tqdm(lst):
            line = line.strip().split(' ')
            dictionary[line[0]] = np.array(line[1:], dtype='float32')
        return dictionary

if __name__ == '__main__':
    glove = GloveMeasurer()
    print(glove.sim(['how', 'do', 'you', 'do'], ['how', 'you', 'do']))
    print('\'ll' in glove.dictionary)
    print(glove.sim(['how', 'do', 'you', 'do'], ['how', 'you', 'do']))