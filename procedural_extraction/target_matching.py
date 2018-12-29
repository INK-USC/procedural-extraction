import re
import logging

import utils
import fuzzy_matching
from procedural_extraction.target_processor import TargetProcessor

log = logging.getLogger(__name__)

def split(lines):
    """
    Split a sentence
    """
    def split_ifthen_cite(splited):
        # split by IF THEN and citation
        
        patt_if = re.compile(r'^\s*(IF)\s+(.+)\s+(THEN)\s+(.+)$')
        patt_cite = re.compile(r'\s*\(([\d, ]+)\)')

        def create_labels(splits, iftype, ifobj, oritext):
            # add one more for final split
            cites = [list(map(int, cite.split(', '))) for cite in splits[1::2]] + [[]]
            assert (len(splits) // 2 + 1) == len(cites), "Wrong dimension: splits %d, cites %d" % (len(splits) // 2 + 1, len(cites))
            return [{
                'oritext': oritext,
                'text': span,
                'iftype': iftype,
                'ifobj': ifobj,
                'cite': cite
            } for (span, cite) in zip(splits[::2], cites)]

        new_splited = list()
        for span in splited:
            text = span['text']
            m = patt_if.search(text)
            if m is None:
                # IF clause no found
                s = patt_cite.split(text)
                # No cite for last one
                new_splited.extend(create_labels(
                    splits=s, 
                    iftype=None, 
                    ifobj=None, 
                    oritext=text))
            else:
                # split if clause by citation -> if phrases
                cur_idx = len(new_splited)
                iftype = m.group(1)
                ifsen = m.group(2)
                s2 = patt_cite.split(ifsen)

                # split then clause by citation -> then phrases
                then_idx = cur_idx + len(s2) // 2 + 1
                thentype = m.group(3)
                thensen = m.group(4)
                s3 = patt_cite.split(thensen)

                # all if phrases: link to all then object
                end_idx = then_idx + len(s3) // 2 + 1
                new_splited.extend(create_labels(
                    splits=s2, 
                    iftype=iftype, 
                    ifobj=list(range(then_idx, end_idx)), 
                    oritext=text))
                assert len(new_splited) == then_idx, "Wrong dimension: expect %d, got %d" % (then_idx, len(new_splited))

                # all then phrases: link to all if object
                new_splited.extend(create_labels(
                    splits=s3, 
                    iftype=thentype, 
                    ifobj=list(range(cur_idx, then_idx)), 
                    oritext=text))
                assert len(new_splited) == end_idx, "Wrong dimension: expect %d, got %d" % (end_idx, len(new_splited))
        return new_splited
    
    def prune(splited):
        # Prune split with empty sentence
        old2new = dict()
        new_splited = list()
        for (idx, span) in enumerate(splited):
            span['text'] = utils.prune(span['text'])
            if span['text'] == '':
                continue
            old2new[idx] = len(new_splited)
            new_splited.append(span)

        # Update ifobj after prune
        for item in new_splited:
            if item['ifobj'] is not None:
                item['ifobj'] = [old2new[obj] for obj in item['ifobj'] if obj in old2new]
        
        return new_splited

    splited = [{
        'text': intext.strip(),
        'iftype': None,
        'ifobj': None,
        'cite': []
    } for intext in lines]

    splited = split_ifthen_cite(splited)
    splited = prune(splited)

    return splited

def retrieve(lines):
    """
    Process protocol file lines
    """

    def add_labels(samples, extractor):
        """
        Add labels from extractor to samples
        """
        new_samples = list()
        for sample in samples:
            new_sample = extractor.process(sample['text'])
            for key in ["iftype", "ifobj", "cite"]:
                new_sample[key] = sample[key]
            new_samples.append(new_sample)
        return new_samples

    def update_next(samples):
        """
        Update next_pos, next_type and next_id to next action if no existing next action by GOTO branch in extractor
        Modify input "samples"
        """
        for idx in range(len(samples) - 1):
            cur = samples[idx]
            nxt = samples[idx+1]
            if cur['next_pos'] is None and cur['cur_pos'][0] == nxt['cur_pos'][0]:
                cur['next_pos'] = nxt['cur_pos']
                cur['next_type'] = nxt['cur_type']
                cur['next_id'] = idx+1
            else:
                cur['next_id'] = None
        samples[-1]['next_id'] = None

    def retrive_positions(samples):
        """
        Regist all positions to their id
        First come first serve
        Modify on samples
        """
        pos2id = dict()
        for (idx, sample) in enumerate(samples):
            pos_str = utils.posstr(sample['cur_pos'])    
            if pos_str not in pos2id:
                pos2id[pos_str] = idx
        
        for (idx, sample) in enumerate(samples):
            if sample['next_id'] is None and sample['next_pos'] is not None:
                pos_str = ''.join(map(str, sample['next_pos']))
                try:
                    samples[idx]['next_id'] = pos2id[pos_str]
                except KeyError:
                    print('Position key %s no found' % pos_str)

    

    spliteds = split(lines)
    extractor = TargetProcessor()
    samples = add_labels(spliteds, extractor)
    update_next(samples)
    retrive_positions(samples)
    
    return samples

def match(samples, src, parser):
    """
    Do fuzzy matching
    """
    args, extra = parser.parse_known_args()

    all_toked_ngrams = src.get_toked_ngrams()
    log.info('%d possible N-grams exists in source file' % len(all_toked_ngrams))

    log.info("Creating matching queries")
    queries = list()
    metas = list()
    for sample in samples:
        cites = sample['cite']
        if args.no_ref or not len(cites):
            src_sens = all_toked_ngrams
        else:
            src_sens = list()
            for cite in set(cites):
                src_sens.extend(src.get_toked_ngrams_line(cite))
        protocol = sample['text']
        toked_candidates = tuple(src_sen['span'] for src_sen in src_sens)
        queries.append((protocol,) + toked_candidates)
        metas.append(src_sens)

    nearest = fuzzy_matching.getNearestMethod(args.method, parser)
    log.info("Finding nearest candidates")
    nearest_idice = nearest(queries)

    for (sample, nearest_idx, meta) in zip(samples, nearest_idice, metas):
        sample['src_matched'] = meta[nearest_idx] if nearest_idx is not None else None

    return samples