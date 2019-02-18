import numpy as np
from tqdm import tqdm

from .dist import register_dist_adaptor
from .measurer_glove import GloveMeasurer
from utils import Tokenizer
from fuzzy_matching.manual_rules import manual_rules

@register_dist_adaptor('embavg')
def embavg_adaptor(parser):
    """
    Glove average embedding
    """
    group = parser.add_argument_group('embavg')
    group.add_argument('--dir_glove', help='Directory of glove embedding', default='embeddings/glove.840B.300d.txt')
    args, extra = parser.parse_known_args()

    glove = GloveMeasurer(args.dir_glove)
    tokenizer = Tokenizer()

    def method(_, queries):
        nearests = list()
        for query in tqdm(queries):
            protocol = query[0]
            toked_protocol = tokenizer.tokenize(protocol[0])
            toked_candidates = [q[0] for q in query[1:]]
            if not len(toked_protocol):
                nearests.append(None)
                continue
            
            min_dist = np.inf
            min_idx = None
            for (idx, toked_can) in enumerate(toked_candidates):
                if not len(toked_can):
                    continue
                dist = -glove.sim(toked_can, toked_protocol)
                dist = manual_rules(toked_can, dist)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            if min_dist > -0.5:
                min_idx = None
            nearests.append(min_idx)

        return nearests

    return method