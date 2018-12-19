import re
import pickle
import time
import argparse
import os.path
import logging as log

import numpy as np

import utils
import fuzzy_matching
from procedural_extraction import SourceProcessor, match, retrieve, split, create_relation_dataset, create_seqlabel_dataset

def main():
    parser = argparse.ArgumentParser(description='Process basic target file processing informations.')
    parser.add_argument('datasetid', metavar='N', type=int, help='Datasets to process')
    parser.add_argument('method', choices=fuzzy_matching.getMethodNames(),  help='Method of fuzzy matching')
    parser.add_argument('--no_ref', action='store_true', help='If not find closest among refer sentences, but globally')
    # data
    parser.add_argument('--dir_data', default='data', help='Dir keeping source data')
    # saving
    parser.add_argument('--dir_seqlabel', default='dataset/seqlabel', help='Dir save seq label dataset')
    parser.add_argument('--dir_relation', default='dataset/relation', help='Dir save phrase pair relation dataset')
    parser.add_argument('--dir_extracted', default='extracted', help='Dir save extracted data')
    # misc
    parser.add_argument("-v", "--verbosity", help="Logging verbosity")

    args, extra = parser.parse_known_args()
    log.basicConfig(level=getattr(log, args.verbosity), handlers=[log.StreamHandler()])
    log.info(args)
    
    ds_idx = args.datasetid
    log.info("Processing dataset %d" % ds_idx)

    path_to_target = os.path.join(args.dir_data, '%02d.tgt.txt' % ds_idx)
    log.info('Reading protocol %d from %s' % (ds_idx, path_to_target))
    with open(path_to_target, 'r') as f:
        lst = f.readlines()

    log.info("Retrieveing samples from protocol file")
    samples = retrieve(lst)

    path_to_src = os.path.join(args.dir_data, '%02d.src.txt' % ds_idx)
    path_to_src_ref = os.path.join(args.dir_data, '%02d.src.ref.txt' % ds_idx)
    log.info("Loading source file %s" % path_to_src)
    src = SourceProcessor(path_to_src, path_to_src_ref)

    log.info("Matching samples to source file")
    samples = match(samples, src, parser)

    log.info("Creating datasets")
    path_to_seqlabel = os.path.join(args.dir_seqlabel, "%02d.part.seqlabel.pkl" % ds_idx)
    create_seqlabel_dataset(samples, src, path_to_seqlabel)
    path_to_relation = os.path.join(args.dir_seqlabel, "%02d.part.relation.tsv" % ds_idx)
    create_relation_dataset(samples, path_to_relation)
    
    for (idx, annotation) in enumerate(samples):
        log.info(idx, annotation)

    path_to_saving = os.path.join(args.dir_extracted, '%02d.tgt.extracted.pkl' % ds_idx)
    log.info("Saving extracted infos to %s" % path_to_saving)
    with open(path_to_saving, 'wb') as f:
        pickle.dump(samples, f)

if __name__ == '__main__':
    main()
