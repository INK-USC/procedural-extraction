import numpy as np
import math

from tqdm import tqdm

from utils import Tokenizer
from .embed_measurer import EmbeddingMeasurer
from .glove_measurer import GloveMeasurer
from .bert_extractor import BertExtractor

_method_builders = dict()

def register_dist_builder(method_name):
    def decorator(func):
        _method_builders[method_name] = func
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
        return wrapper
    return decorator

def getNearestMethod(method_name, parser):
    """
    all candidates toked
    all protocol untoked
    input:
    [
        (protocol, candidate1, candidate2, ...),
        (protocol, candidate1, candidate2, ...),
        (protocol, candidate1, candidate2, ...),
        ...
    ]
    output:
    [
        nearest_idx1,
        nearest_idx2,
        nearest_idx3,
        ...
    ]
    """
    return _method_builders[method_name](parser)

def getMethodNames():
    return list(_method_builders.keys())

@register_dist_builder('embavg')
def embavg_builder(parser):
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

@register_dist_builder('bert')
def bert_builder(parser):
    """
    Glove average embedding
    """
    bert = BertExtractor(parser)

    def method(queries):
        sentences = set()
        for query in queries:
            sentences.add(query[0])
            sentences.update(list(map(' '.join, query[1:])))
        sentences = list(sentences)

        sen_embs = bert.extract(sentences)
        sen2emb = dict()
        for (sen_emb, sentence) in zip(sen_embs, sentences):
            sen2emb[sentence] = sen_emb
        
        measurer = EmbeddingMeasurer(sen2emb)
        
        nearests = list()
        for query in tqdm(queries):
            protocol = query[0]
            toked_candidates = query[1:]
            candidates = list(map(' '.join, query[1:]))
            min_dist = np.inf
            min_idx = None
            for (idx, toked_can) in enumerate(candidates):
                dist = -measurer.sim(toked_can, protocol)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
    
            nearests.append(min_idx)

        return nearests

    return method
        