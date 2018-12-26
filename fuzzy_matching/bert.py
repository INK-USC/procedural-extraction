import numpy as np
from tqdm import tqdm

from .dist_methods import register_dist_adaptor
from .embed_measurer import EmbeddingMeasurer
from models.bert_classifier import BertClassifier

@register_dist_adaptor('bert')
def bert_adaptor(parser):
    """
    Glove average embedding
    """
    bert = BertClassifier(parser)

    def method(queries):
        pairs = list()
        belongs = list()
        for (i, query) in enumerate(queries):
            protocol = query[0]
            candidates = query[1:]
            for can in candidates:
                pairs.append((protocol, ' '.join(can)))
                belongs.append(i)

        dists = bert.predict(pairs)

        nearests = list()
        for (i, query) in enumerate(queries):
            nearests.append(np.argmin(dists[belongs == i]))

        return nearests

    return method