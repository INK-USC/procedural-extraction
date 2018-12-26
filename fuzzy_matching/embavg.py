import numpy as np
from tqdm import tqdm

from .dist_methods import register_dist_adaptor
from .glove_measurer import GloveMeasurer
from utils import Tokenizer

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

    def method(queries):
        nearests = list()
        for query in tqdm(queries):
            toked_protocol = tokenizer.tokenize(query[0])
            toked_candidates = query[1:]
            if not len(toked_protocol):
                nearests.append(None)
                continue
            
            min_dist = np.inf
            min_idx = None
            for (idx, toked_can) in enumerate(toked_candidates):
                if not len(toked_can):
                    continue
                dist = -glove.sim(toked_can, toked_protocol)
                if toked_can[0] in ['and', 'or', ',']:
                    dist *= 0.5
                if toked_can[1] in [',.']:
                    dist *= 0.5
                if toked_can[-1] in ['the', 'of', ',', '.', ':', 'to', 'and', 'in', 'this', 'that', 'or']:
                    dist *= 0.5
                if toked_can[-2] in [',.']:
                    dist *= 0.5
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
    
            nearests.append(min_idx)

        return nearests

    return method