import numpy as np
from tqdm import tqdm
import json

from .dist import register_dist_adaptor
from .measurer_glove import GloveMeasurer
from utils import Tokenizer
from fuzzy_matching.manual_rules import manual_rules

@register_dist_adaptor('manual')
def manual_adaptor(parser):
    """
    manual matching
    """
    group = parser.add_argument_group('manuals')
    group.add_argument('--dir_glove', help='Directory of glove embedding', default='embeddings/glove.840B.300d.txt')
    group.add_argument('--dump', action='store_true', help='Output manual input sheet or input manual annotations')
    args, extra = parser.parse_known_args()

    def method(_, queries):
        if args.dump == True:
            jsobj = []
            for q in queries:
                o = {}
                o['protocol'] = q[0][0]
                o['zcandidates'] = [' '.join(i[0]) for i in q[1:]]
                jsobj.append(o)
            json.dump(jsobj, open('fuzzy_matching/inputsheet.json', 'w'), indent=4, sort_keys=True)
            raise ValueError("fuzzy_matching/inputsheet.json dumped. Please choose nearest sentences and rename to answer.json")
        else:
            obj = json.load(open('fuzzy_matching/answer.json', 'r'))
            nearest = []
            oidx = 0
            for q in queries:
                if obj[oidx]['protocol'] != q[0][0]:
                    nearest.append(None)
                    continue
                o = obj[oidx]
                oidx += 1
                if not len(o['zcandidates']):
                    nearest.append(None)
                    continue
                answer = o['zcandidates'][0]
                matched_idx = None
                for (idx, c) in enumerate(q[1:]):
                    if ' '.join(c[0]) == answer:
                        matched_idx = idx
                        break
                if matched_idx is None:
                    raise ValueError('unfound key found')
                nearest.append(matched_idx)
            return nearest

    return method