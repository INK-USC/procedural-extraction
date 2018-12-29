import numpy as np
from tqdm import tqdm

from .dist import register_dist_adaptor
from .measurer_embed import EmbeddingMeasurer, L2norm
from models.bert_extractor import BertExtractor

@register_dist_adaptor('ex-bert')
def extracted_bert_adaptor(parser):
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
            sen2emb[sentence] = L2norm(sen_emb)
        
        measurer = EmbeddingMeasurer(sen2emb, False)
        
        nearests = list()
        for query in tqdm(queries):
            protocol = query[0]
            candidates = list(map(' '.join, query[1:]))
            min_dist = np.inf
            min_idx = None
            for (idx, can) in enumerate(candidates):
                dist = -measurer.sim(can, protocol)
                if can[0] in [',','.']:
                    dist *= 0.5
                if can[-1] in [',','.']:
                    dist *= 0.5
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            if min_dist > -0.5:
                min_idx = None
            nearests.append(min_idx)

        return nearests

    return method