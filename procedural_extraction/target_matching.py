import re
import logging

import utils
import json
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
            print(samples[idx])
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

def match(samples, src, parser, eval=False):
    """
    Do fuzzy matching
    """
    args, _ = parser.parse_known_args()

    all_toked_ngrams = src.get_toked_ngrams()
    log.info('%d possible N-grams exists in source file' % len(all_toked_ngrams))

    log.info("Creating matching queries")
    queries = list()
    metas = list()
    newsamples = list()
    candidates_list = list()
    ori2new = dict()
    n_i = 0
    for (i, sample) in enumerate(samples):
        cites = sample['cite']
        if args.no_ref:
            src_sens = all_toked_ngrams
        elif not len(cites):
            ori2new[i] = None
            continue
        else:
            src_sens = list()
            for cite in set(cites):
                src_sens.extend(src.get_toked_ngrams_line(cite))
        protocol = sample['text']
        metas.append(src_sens)
        candidates = [(src_sen['span'], src_sen['src_sens_id'], src_sen['start'], src_sen['K']) for src_sen in src_sens]
        queries.append([(protocol, )] + candidates)
        newsamples.append(sample)
        candidates_list.append(candidates)
        ori2new[i] = n_i
        n_i += 1

    nearest = fuzzy_matching.get_nearest_method(args.method, parser)
    log.info("Finding nearest candidates")
    nearest_idice = nearest(src.src_sens, queries)

    print('samples', len(newsamples), 'metas', len(metas))
    for (sample, nearest_idx, meta) in zip(newsamples, nearest_idice, metas):
        sample['src_matched'] = meta[nearest_idx] if nearest_idx is not None else None

    if args.eval:
        obj = json.load(open('answer.json', 'r'))
        totm, tot = 0, 0
        BIN_LEN = 20
        binm, bint = [0] * BIN_LEN, [0] * BIN_LEN
        def addbin(length, m=False):
            if length >= BIN_LEN:
                length = BIN_LEN - 1
            if m:
                binm[length] += 1
            else:
                bint[length] += 1
            

        ttp, ttn, tfp, tfn = 0, 0, 0, 0
        for q, o, candidates in zip(newsamples, obj, candidates_list):
            if not len(o['zcandidates']):
                continue
            tot += 1
            answer = o['zcandidates'][0]
            alen = len(o['zcandidates'][0].split(' '))
            addbin(alen)
            if q['src_matched'] is not None and ' '.join(q['src_matched']['span']) == answer:
                totm += 1
                addbin(alen, True)
            mid, mst, mK = None, None, None
            for c in candidates:
                if ' '.join(c[0]) == answer:
                    mid, mst, mK, = c[1:]
            if mid is None:
                raise ValueError('unfound key found')
            if q['src_matched'] is None or mid != q['src_matched']['src_sens_id']:
                ll = len(src.src_sens[mid])
                tfn += mK / ll
                ttn += (len(src.src_sens[mid]) - mK) / ll

            else:
                ll = len(src.src_sens[mid])
                logits1 = [0] * ll
                logits2 = [0] * ll
                for i in range(mst, mst + mK):
                    logits1[i] = 1
                for i in range(q['src_matched']['start'], q['src_matched']['K']):
                    logits2[i] = 1
                for a, b in zip(logits1, logits2):
                    if a == 0 and b == 0:
                        ttn += 1.0/ll
                    elif a == 0 and b == 1:
                        tfp += 1.0/ll
                    elif a == 1 and b == 0:
                        tfn += 1.0/ll
                    elif a == 1 and b == 1:
                        ttp += 1.0/ll

        print("Eval result, mention level:", totm, '/', tot, 'of samples matched')
        precision = 1.0 * ttp / (ttp + tfp) 
        recall = 1.0 * ttp / (ttp + tfn)
        print("Eval result, token level: precision", precision, 'recall', recall, 'f1', precision * recall * 2.0 / (precision + recall))
        print(binm, '/', bint)

    return newsamples, ori2new