import logging
import utils
import argparse
from itertools import product

import numpy as np
import os

from .dsbuilder import register_dsbuilder

log = logging.getLogger(__name__)

@register_dsbuilder('relation')
def builder_relation_dataset(parser: argparse.ArgumentParser):
    parser.add_argument('--balance', default=1.0, metavar='portion', type=float, help='label balancing portion, -1.0 to disable balancing, N balance pos : neg samples by 1 : N')
    parser.add_argument('--part', default=8, metavar='N', type=int, help='dataset portion, divide dataset train: dev: test by N-2:1:1')
    parser.add_argument('--path', default='dataset/relation', metavar='path_to_dir', help='dir to save the dataset')
    parser.add_argument('--format', default='tsv', choice=['tsv'], help='format to save')

    args = parser.parse_args()

    assert args.part >= 3, '--part should >= 3'

    def _method(sample_sets):
        dataset = {
            'next': [],
            'if': [],
            'none': []
        }

        for samples in sample_sets:
            for ((idx, sample), (idx2, sample2)) in product(enumerate(samples), enumerate(samples)):
                if sample['src_matched'] is None or sample2['src_matched'] is None or idx == idx2:
                    continue
                text1 = ' '.join(sample['src_matched']['span'])
                text2 = ' '.join(sample2['src_matched']['span'])
                if idx2 == sample['next_id']:
                    dataset['next'].append((text1, text2, 'next'))
                elif sample['iftype'] == 'THEN' and idx2 in sample['ifobj']:
                    dataset['if'].append((text1, text2, 'if'))
                else:
                    dataset['none'].append((text1, text2, 'none'))

        min_sz = min(map(len, dataset.values()))
        #clean files
        for subset in ['train', 'dev', 'test']:
            open(os.path.join(args.path, '%s.tsv' % subset), 'w').close()
        for (relation, triplets) in dataset.items():
            np.random.shuffle(triplets)
            if args.balance != -1 and relation == 'none':
                triplets = triplets[:min_sz * args.balance]
            part_sz = int(len(triplets)/args.part)

            with open(os.path.join(args.path, 'train.tsv'), 'a') as f:
                for triplet in triplets[:-2 * part_sz]:
                    f.write("%s\t%s\t%s\n" % triplet)
            with open(os.path.join(args.path, 'dev.tsv'), 'a') as f:
                for triplet in triplets[-2 * part_sz : - part_sz]:
                    f.write("%s\t%s\t%s\n" % triplet)
            with open(os.path.join(args.path, 'test.tsv'), 'a') as f:
                for triplet in triplets[- part_sz : ]:
                    f.write("%s\t%s\t%s\n" % triplet)
                    
    return _method
        
    