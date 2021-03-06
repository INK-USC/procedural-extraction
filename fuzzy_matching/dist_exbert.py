import numpy as np
from tqdm import tqdm

from .dist import register_dist_adaptor
from .measurer_embed import EmbeddingMeasurer, L2norm
from models.bert_extractor import BertExtractor
from fuzzy_matching.manual_rules import manual_rules

@register_dist_adaptor('ex-bert')
def extracted_bert_adaptor(parser):
    """
    Bert embedding
    """
    parser.add_argument('--mask', action='store_true', help='if apply mask to extracted embedding')
    args, extra = parser.parse_known_args()
    bert = BertExtractor(parser)

    def method(src_sens, queries):
        """
        sentences: token sentences
        queries:  
        queries_meta: []
        """
        embed_mapping = dict()
        if args.mask:
            sen_embs = bert.extract([' '.join(sen) for sen in src_sens])
            for (idx, query) in enumerate(queries):
                for (candidate, sen_id, start, K) in query[1:]:
                    phrase_embs = sen_embs[sen_id][start+1: start+K+1]
                    if phrase_embs.shape[0] != K:
                        print(sen_embs[sen_id].shape[0])
                        print(sen_id, start, K, len(src_sens[sen_id]))
                        raise ValueError("mismatch token dimension")
                    query_emb = np.average(phrase_embs, axis=0)
                    embed_mapping[' '.join(candidate)] = query_emb
            protocol_embs = bert.extract([query[0][0] for query in queries])
            for (idx, emb) in enumerate(protocol_embs):
                embed_mapping[queries[idx][0][0]] = np.average(emb, axis=0)
            embed_mapping = {k: L2norm(v) for k, v in embed_mapping.items()}
        else:
            sentences = set()
            for query in queries:
                sentences.add(query[0][0])
                sentences.update(list([' '.join(q[0]) for q in query[1:]]))
            sentences = list(sentences)

            sen_embs = bert.extract(sentences)
            for (sen_emb, sentence) in zip(sen_embs, sentences):
                embed_mapping[sentence] = L2norm(np.average(sen_emb, axis=0))
            
        measurer = EmbeddingMeasurer(embed_mapping, False)
        
        nearests = list()
        for query in tqdm(queries):
            protocol = query[0][0]
            if len(protocol) == 0:
                nearests.append(None)
                continue
            candidates = list(map(' '.join, [q[0] for q in query[1:]]))
            min_dist = np.inf
            min_idx = None
            for (idx, can) in enumerate(candidates):
                dist = -measurer.sim(can, protocol)
                dist = manual_rules(can, dist)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            if min_dist > -0.5:
                min_idx = None
            nearests.append(min_idx)

        return nearests

    return method