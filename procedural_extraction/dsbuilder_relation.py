import logging
import utils
import pickle
import pprint
import argparse
from itertools import product
import json

import numpy as np
import os

from .source_processor import SourceProcessor
from .dsbuilder import register_dsbuilder
from utils.input_context_sample import ISen, IExample

log = logging.getLogger(__name__)

@register_dsbuilder('relation')
def builder_relation_dataset(parser: argparse.ArgumentParser):
    parser.add_argument('--ratio_none', default=6.0, metavar='6.0', type=float, help='label balancing portion for label None, negative number to disable balancing')
    parser.add_argument('--ratio_next', default=3.0, metavar='3.0', type=float, help='label balancing portion for label Next, negative number to disable balancing')
    parser.add_argument('--part', default=8, metavar='N', type=int, help='dataset portion, divide dataset train: dev: test by N-2:1:1. don\'t part if < 1')
    parser.add_argument('--path', default='dataset/relation', metavar='path_to_dir', help='dir to save the dataset')
    parser.add_argument('--k_neighbour', default=1, metavar='K', type=int, help='range of context sentences')
    parser.add_argument('--output', action="store_true", help='shows example data')
    parser.add_argument('--no_context', action="store_true", help='w/o context info')
    parser.add_argument('--seed', default=42)

    args = parser.parse_args()

    assert args.part >= 3, '--part should >= 3'
    np.random.seed(args.seed)

    def _method(sample_sets):
        dataset = {
            'next': list(),
            'if': list(),
            'none': list()
        }
        k = args.k_neighbour
        if args.no_context:
            k = 0
        print("Creating dataset with %d neighbour and no context is %s" % (k, str(args.no_context)))
        
        def movone(x):
            if x < 0:
                return x - 1 
            elif x > 0:
                return x + 1
            else:
                return x

        def build_block(src, meta, alter_meta):
            id = meta['src_sens_id']
            alter_id = alter_meta['src_sens_id']
            context_l = [ISen(
                    text=' '.join(src[id][:meta['start']]),
                    offset=-1, 
                    alter_offset=movone(id - alter_id)
                )] if not args.no_context else []
            context_r = [ISen(
                    text=' '.join(src[id][meta['start'] + meta['K']:]), 
                    offset=1, 
                    alter_offset=movone(id - alter_id)
                )] if not args.no_context else []

            return [ISen(
                    text=' '.join(src[i]), 
                    offset=movone(i-id), 
                    alter_offset=movone(i-alter_id)
                )  for i in range(max(0, id-k), id)] + \
                context_l + \
                [ISen(
                    text=' '.join(meta['span']), 
                    offset=0, 
                    alter_offset=movone(id - alter_id)
                )] + \
                context_r + \
                [ISen(
                    text=' '.join(src[i]), 
                    offset=movone(i-id), 
                    alter_offset=movone(i-alter_id)
                ) for i in range(id+1, min(len(src), id+k+1))]

        no_matched = set()
        for (dsid, samples) in sample_sets:
            path_src = utils.path.src(args.dir_data, dsid)
            path_src_ref = utils.path.src_ref(args.dir_data, dsid)
            log.info("Loading source file %s" % path_src)
            src = SourceProcessor(path_src, path_src_ref)

            for ((idx, sample), (idx2, sample2)) in product(enumerate(samples), enumerate(samples)):
                meta1, meta2 = sample['src_matched'], sample2['src_matched']
                if meta1 is None:
                    no_matched.add(str(dsid) + '_' + str(idx))
                    continue
                if meta2 is None or idx == idx2:
                    continue
                dat = IExample(
                    build_block(src.src_sens, meta1, meta2),
                    build_block(src.src_sens, meta2, meta1),
                )
                if idx2 == sample['next_id']:
                    dat.label = 'next'
                    dataset['next'].append(dat)
                elif sample['iftype'] == 'THEN' and idx2 in sample['ifobj']:
                    dat.label = 'if'
                    dataset['if'].append(dat)
                else:
                    dat.label = 'none'
                    dataset['none'].append(dat)

        for k, v in dataset.items():
            print(k, 'have', len(v), 'samples')
        print('sample without src_matched:', no_matched)

        splited_set = {
            'train': list(), # train set
            'dev': list(), # dev set
            'test': list(), # test set
            'nonsplit': list(), # dataset without division
            'full': list() # dataset without division or sampling
        }

        fewest = len(dataset['if'])
        for (relation, triplets) in dataset.items():
            np.random.shuffle(triplets)
            splited_set['full'].extend(triplets)
            if relation == 'none' and args.ratio_none >= 0:
                triplets = triplets[:int(fewest * args.ratio_none)]
            elif relation == 'next' and args.ratio_next >= 0:
                triplets = triplets[:int(fewest * args.ratio_next)]
            part_sz = int(len(triplets)/args.part)          
            splited_set['train'].extend(triplets[:-2 * part_sz])
            splited_set['dev'].extend(triplets[-2 * part_sz : - part_sz])
            splited_set['test'].extend(triplets[- part_sz : ])
            splited_set['nonsplit'].extend(triplets)

        print('total', len(splited_set['nonsplit']))

        if not os.path.exists(args.path):
            print("creating", args.path)
            os.makedirs(args.path)

        for (key, triplets) in splited_set.items():
            with open(os.path.join(args.path, key+'.pkl'), 'wb') as f:
                pickle.dump(triplets, f)
            with open(os.path.join(args.path, key+'.json'), 'w') as f:
                json_lines = list()
                for (idx, example) in enumerate(triplets):
                    text_left = ""
                    text_right = ""
                    for sen in example.left:
                        text_left += sen.text
                    for sen in example.right:
                        text_right += sen.text
                    json_lines.append(json.dumps({
                        'text_a': text_left,
                        'text_b': text_right,
                        'label': example.label,
                        'pair_id': idx
                    }))
                f.write('\n'.join(json_lines))
                    
    return _method