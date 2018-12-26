import re
import pickle
import time
import argparse
import os.path
import logging

import numpy as np

import utils
import fuzzy_matching
from procedural_extraction.source_processor import SourceProcessor
from procedural_extraction.target_matching import split, retrieve, match
from procedural_extraction.create_dataset import create_seqlabel_dataset, create_relation_dataset

log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='process basic target file processing informations.')
    parser.add_argument('datasetid', metavar='N', type=int, help='dataset id to process(0-99)')
    parser.add_argument('method', choices=fuzzy_matching.getMethodNames(),  help='method of fuzzy matching')
    parser.add_argument('--no_ref', action='store_true', help='don\'t find closest among refer sentences, but globally')
    # data
    parser.add_argument('--dir_data', default='data', help='specify dir holding source data')
    # saving
    parser.add_argument('--dir_seqlabel', default='dataset/seqlabel', help='specify dir save seq label dataset')
    parser.add_argument('--dir_relation', default='dataset/relation', help='specify dir save phrase pair relation dataset')
    parser.add_argument('--dir_extracted', default='extracted', help='specify dir save extracted data')
    # misc
    parser.add_argument("--verbosity", help="logging verbosity", default="INFO")
    parser.add_argument("--src_retok", action="store_true", help="re-tokenize source file, ignoring existing cache")
    parser.add_argument("--output", action="store_true", help="show extracted results to stdout")

    args, extra = parser.parse_known_args()
    logging.basicConfig(level=getattr(logging, args.verbosity), handlers=[logging.StreamHandler()])
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
    src = SourceProcessor(path_to_src, path_to_src_ref, args.src_retok)

    log.info("Matching samples to source file")
    samples = match(samples, src, parser)

    if args.output:
        for (idx, annotation) in enumerate(samples):
            print(idx, annotation)

    log.info("Creating datasets")
    path_to_seqlabel = os.path.join(args.dir_seqlabel, "%02d.part.seqlabel.pkl" % ds_idx)
    create_seqlabel_dataset(samples, src, path_to_seqlabel)
    path_to_relation = os.path.join(args.dir_relation, "%02d.part.relation.pkl" % ds_idx)
    create_relation_dataset(samples, path_to_relation)

    path_to_saving = os.path.join(args.dir_extracted, '%02d.tgt.extracted.pkl' % ds_idx)
    log.info("Saving extracted infos to %s" % path_to_saving)
    with open(path_to_saving, 'wb') as f:
        pickle.dump(samples, f)

if __name__ == '__main__':
    main()
