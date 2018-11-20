from rouge import Rouge 
from embed_measure import GloveMeasurer
import numpy as np
import math

class DistCalculator(object):
    def __init__ (self):
        self.kv = {
            'embavg': self.embavg,
        }
        self.max_distance = 10000
        self.glove = GloveMeasurer()

    def calc_dist(self, sen_ori, sen_hyp, method='embavg'):
        if len(sen_ori) == 0 or len(sen_hyp) == 0:
            return self.max_distance
        sen_ori = sen_ori.lower()
        sen_hyp = sen_hyp.lower()
        return self.kv[method](sen_ori, sen_hyp)
    
    def calc_dist_multi_one(self, sen_multi, sen_one, method='embavg'):
        """
        return index of best origin
        """
        sen_one = sen_one.lower()
        sen_multi = [sen.lower() for sen in sen_multi]
        if len(sen_multi):
            dists = self.kv[method](sen_multi, sen_one)
            if np.min(dists) < self.max_distance:
                return np.argmin(dists)
        return None

    def embavg(self, sen_multi, sen_one):
        """
        Hyp: protocol
        Ori: source spans
        """
        toked_sen_one = tokenize(sen_one)
        if not len(toked_sen_one):
            # unable to tokenize
            return [self.max_distance]
        dists = list()
        for sen in sen_multi:
            toked_sen = sen.split(' ')
            dist = -self.glove.sim(toked_sen, toked_sen_one)
            if toked_sen[0] in ['and', 'or', ',']:
                dist *= 0.8
            if toked_sen[1] in [',']:
                dist *= 0.8
            if toked_sen[-1] in ['the', 'of', ',', '.', ':', 'to', 'and', 'in', 'this', 'that', 'or']:
                dist *= 0.8
            if toked_sen[-2] in [',']:
                dist *= 0.8
            dists.append(self.max_distance if dist > -0.1 else dist)
        return dists