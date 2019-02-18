"""
Author: Junyi Du
oct. 8 2018
"""

import numpy as np
from tqdm import tqdm
import os.path
from typing import List

def L2norm(embed):
    norm = (embed @ embed)**0.5
    return embed / norm

def dot(emb1, emb2):
    """
    return distance between two words
    """
    return emb1 @ emb2

def mle(emb1, emb2):
    return -((emb1 - emb2) @ (emb1 - emb2))

class EmbeddingMeasurer(object):
    def __init__(self, preset_memory=None, need_join=True, measure='dot'):
        self.func = {'dot': dot, 'mle': mle}[measure]
        self.need_join = need_join
        if preset_memory is None:
            self.memory = {}
        else:
            self.memory = preset_memory

    def sen_emb(self, sen):
        """
        return average sentence embedding of given sentence
        """
        key = ' '.join(sen) if self.need_join else sen
        if key not in self.memory:
            self.memory[key] = L2norm(self.vanilla_sen_emb(sen))
        return self.memory[key]

    def vanilla_sen_emb(self, sen) -> np.array:
        raise NotImplementedError("!")
        
    def sim(self, sen1, sen2):
        """
        return similarity between two sentences
        """
        return self.func(self.sen_emb(sen1), self.sen_emb(sen2))

    
