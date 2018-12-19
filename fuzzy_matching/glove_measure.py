import pickle
import os
import logging as log

from tqdm import tqdm
import numpy as np

from .embed_measure import EmbeddingMeasurer

class GloveMeasurer(EmbeddingMeasurer):
    def __init__(self, glove_path):
        super().__init__()

        glove_pkl_path = glove_path + '.pkl'
        if os.path.isfile(glove_pkl_path):
            log.info('Converted Glove pickle at {} found, loading'.format(glove_pkl_path))
            with open(glove_pkl_path, 'rb') as f:
                self.dictionary = pickle.load(f)
        else:
            log.info('No converted Glove pickle found')
            self.dictionary = self._load_glove(glove_path)
            log.info('Saving converted Glove pickle to {}'.format(glove_pkl_path))
            with open(glove_pkl_path, 'wb') as f:
                pickle.dump(self.dictionary, f)
        self.glove_shape = self.dictionary['glove'].shape

    def _load_glove(self, glove_path):
        log.info('Reading Glove word vectors from {}'.format(glove_path))
        with open(glove_path, 'r') as f:
            lst = f.readlines()

        dictionary = {}

        log.info('Converting Glove word vectors')
        for line in tqdm(lst):
            line = line.strip().split(' ')
            dictionary[line[0]] = np.array(line[1:], dtype='float32')
        return dictionary
        
    def __getitem__(self, item) -> np.array:
        """
        entry of all query
        no L2 norm
        """
        word = item.lower()
        if word not in self.dictionary:
            self.dictionary[word] = np.random.normal(size=self.glove_shape)
        word_vec = self.dictionary[word]
        return word_vec

    def vanilla_sen_emb(self, sen) -> np.array:
        return np.average(list(map(self.__getitem__, sen)), axis=0)

if __name__ == '__main__':
    glove = GloveMeasurer('embeddings/glove.840B.300d.txt')
    print(glove.sim(['how', 'do', 'you', 'do'], ['how', 'you', 'do']))
    print('\'ll' in glove.dictionary)
    print(glove.sim(['how', 'do', 'you', 'do'], ['how', 'you', 'do']))