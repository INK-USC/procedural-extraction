"""
Author: Junyi Du
oct. 8 2018
"""

import numpy as np
from tqdm import tqdm
import os.path
from typing import List

class EmbeddingMeasurer(object):
    def __init__(self, preset_memory=None):
        if preset_memory is None:
            self.memory = {}
        else:
            self.memory = preset_memory

    def sen_emb(self, sen) -> np.array:
        """
        return average sentence embedding of given sentence
        """
        key = ' '.join(sen)
        if key not in self.memory:
            self.memory[key] = self.L2norm(self.vanilla_sen_emb(sen))
        return self.memory[key]

    def vanilla_sen_emb(self, sen) -> np.array:
        raise NotImplementedError("!")
        
    def sim(self, sen1, sen2) -> float:
        """
        return similarity between two sentences
        """
        return self.dot(self.sen_emb(sen1), self.sen_emb(sen2))

    @staticmethod
    def L2norm(embed):
        norm = (embed @ embed)**0.5
        return embed / norm

    @staticmethod
    def dot(emb1, emb2):
        """
        return distance between two words
        """
        return emb1 @ emb2
