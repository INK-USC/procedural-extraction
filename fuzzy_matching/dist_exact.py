import numpy as np
from tqdm import tqdm

from .dist import register_dist_adaptor
from .measurer_glove import GloveMeasurer
from utils import Tokenizer
from fuzzy_matching.manual_rules import manual_rules

@register_dist_adaptor('exact')
def exact_adaptor(parser):
    """
    exact matching
    """
    tokenizer = Tokenizer()

    def method(_, queries):
        nearests = list()
        for query in tqdm(queries):
            protocol = query[0]
            toked_protocol = tokenizer.tokenize(protocol[0])
            catprotocol = ' '.join(toked_protocol)
            toked_candidates = [q[0] for q in query[1:]]
            if not len(toked_protocol):
                nearests.append(None)
                continue
            
            max_matched = 0
            max_idx = None
            for (idx, toked_can) in enumerate(toked_candidates):
                if ' '.join(toked_can) in catprotocol and len(toked_can) > max_matched:
                    max_idx = idx
                    max_matched = len(toked_can)
            nearests.append(max_idx)

        return nearests

    return method