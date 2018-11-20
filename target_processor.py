import re
import pickle
import time
import argparse
import os.path

import numpy as np

import utils
from fuzzy_matching import DistCalculator
from source_processor import SourceTranscriptProcessor

dist_calculator = DistCalculator()

class ProtocolExtractor(object):
    """
    Extract structured infos from natural text
    Init class for one document
    Invoke process for each line
    """
    def __init__(self):
        """
        Init for doc
        """
        self.patt_step = re.compile(r'^\s*(Task|Step)\s+([^:]+):\s+')
        self.patt_branch = re.compile(r'^\s*(BRANCH)\s+([^:]+):\s+')
        self.patt_goto = re.compile(r'(GOTO)\s+(Task|Step|BRANCH)\s+([^\s]+)')

        self.cur_pos = ['S']
        self.cur_type = None

    def text(self, intext):
        """
        Init for each splited sentence
        """
        self.ori_text = intext
        self.line_text = intext
        self.next_type = None
        self.next_pos = None
        return self

    def step(self):
        """
        Extract task & step infos
        "Task A.3.1: blablabla" -> ['A', 3, 1]
        """
        text = self.line_text
        patt = self.patt_step
        m = patt.search(text)
        if m is None:
            # If nothing found, follow previous position
            pass
        else:
            # else, use current position
            self.cur_type = m.group(1)
            self.cur_pos = utils.convert_int(m.group(2).split('.'))
            self.line_text = patt.sub('', text)
        return self

    def branch(self):
        """
        Extract Branch infos
        "BRANCH A: blablabla" -> ['A', 3, 1]
        """
        text = self.line_text
        patt = self.patt_branch
        m = patt.search(text)
        if m is None:
            # If nothing found, follow previous position
            pass
        else:
            # else, use current position
            self.cur_type = m.group(1)
            self.cur_pos = m.group(2)
            self.line_text = patt.sub('', text)
        return self

    def goto(self):
        """
        Extract goto infos
        "GOTO BRANCH A"
        """
        text = self.line_text
        patt = self.patt_goto
        m = patt.search(text)
        if m is None:
            # If nothing found, do nothing
            pass
        else:
            # else, update result
            self.cur_type = m.group(1)
            self.next_type = m.group(2)
            self.next_pos = m.group(3)
            self.line_text = patt.sub('', text)
        return self

    def getResult(self):
        return {
            'ori_text': self.ori_text,
            'text': self.line_text,
            'cur_pos': self.cur_pos,
            'cur_type': self.cur_type,
            'next_type': self.next_type,
            'next_pos': self.next_pos
        }

    def process(self, text):
        """
        Process a sub sentence
        """
        res = self.text(text).step().branch().goto().getResult()
        return res

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
    extractor = ProtocolExtractor()
    samples = add_labels(spliteds, extractor)
    update_next(samples)
    retrive_positions(samples)
    
    return samples

def match(samples, src, args):
    all_ngrams = src.get_ngrams()
    if args.method == 'embbert':
        bert.prehot(all_ngrams)
        bert.prehot([sample['text'] for sample in samples])
    for (idx, sample) in enumerate(samples):
        text = sample['text']
        cites = sample['cite']
        if args.no_ref or not len(cites):
            src_sens = all_ngrams
        else:
            sen_lines = set()
            for cite in cites:
                sen_lines.add(cite)
            src_sens = list()
            for line in sen_lines:
                src_sens.extend(src.get_ngrams_line(line))

        best_src_sens_idx = dist_calculator.calc_dist_multi_one([src_sen['span'] for src_sen in src_sens], text, args.method)

        if best_src_sens_idx is not None:
            sample['src_matched'] = src_sens[best_src_sens_idx]
        else:
            sample['src_matched'] = None

    return samples

def create_dataset(samples, src):
    for sample in samples:
        matched = sample['src_matched']
        text = sample['text']
        if matched is not None:
            src.add_matched_ngram(matched, text)
    src.dump_dataset()

def process(lines, src, args):
    print("Retrieveing samples from protocol file")
    samples = retrieve(lines)
    print("Matching samples to source file")
    samples = match(samples, src, args)
    print("Creating dataset")
    create_dataset(samples, src)
    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('datasetid', metavar='N', type=int,
                        help='Datasets to process')
    parser.add_argument('method', choices=['embavg', 'embbert'])
    parser.add_argument('--no-ref', action='store_true', help='If not find closest among refer sentences')
    args = parser.parse_args()
    ds_idx = args.datasetid

    path_to_saving_dir = 'pkls/{:02d}.tgt.extracted.pkl'.format(ds_idx)
    with open('data/{:02d}.tgt.txt'.format(ds_idx), 'r') as f:
        lst = f.readlines()

    print('dataset {}'.format(ds_idx))
    annotations = process(
            lines=lst, 
            src=SourceTranscriptProcessor(
                'data/%02d.src.txt' % ds_idx, 
                'data/%02d.src.ref.txt' % ds_idx),
            args=args
        )
    for (idx, annotation) in enumerate(annotations):
        print(idx, annotation)

    with open(path_to_saving_dir, 'wb') as f:
        pickle.dump(annotations, f)
