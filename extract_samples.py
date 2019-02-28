import re
import pickle
import time
import argparse
import os.path
import logging
import json

import numpy as np

import utils
import fuzzy_matching
from procedural_extraction.source_processor import SourceProcessor
from procedural_extraction.target_matching import split, retrieve, match

log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='process basic target file processing informations.')
    parser.add_argument('datasetid', metavar='N', type=int, help='dataset id to process(0-99)')
    parser.add_argument('method', choices=fuzzy_matching.get_method_names(),  help='method of fuzzy matching')
    parser.add_argument('--no_ref', action='store_true', help='don\'t find closest among refer sentences, but globally')
    # data
    parser.add_argument('--dir_data', default='data', help='specify dir holding source data')
    parser.add_argument('--dir_extracted', default='extracted', help='specify dir to save extracted data')
    # misc
    parser.add_argument("--verbosity", help="logging verbosity", default="INFO")
    parser.add_argument("--src_retok", action="store_true", help="re-tokenize source file, ignoring existing cache")
    parser.add_argument("--output", action="store_true", help="show extracted results to stdout")
    parser.add_argument("--eval", action='store_true', help="evaluate base on manual annotation")
    args, extra = parser.parse_known_args()
    logging.basicConfig(level=getattr(logging, args.verbosity), handlers=[logging.StreamHandler()])
    log.info(args)
    
    ds_idx = args.datasetid
    log.info("Processing dataset %d" % ds_idx)
    path_tgt = utils.path.tgt(args.dir_data, args.datasetid)
    log.info('Reading protocol %d from %s' % (ds_idx, path_tgt))
    with open(path_tgt, 'r') as f:
        lst = f.readlines()

    log.info("Retrieveing samples from protocol file")
    samples = retrieve(lst)

    path_src = utils.path.src(args.dir_data, args.datasetid)
    path_src_ref = utils.path.src_ref(args.dir_data, args.datasetid)
    log.info("Loading source file %s" % path_src)
    src = SourceProcessor(path_src, path_src_ref, args.src_retok)

    log.info("Matching samples to source file")
    samples, ori2mis = match(samples, src, parser, args.eval)

    for sample in samples:
        if sample['next_id'] is not None:
            sample['next_id'] = ori2mis[sample['next_id']]
        if sample['ifobj'] is not None:
            sample['ifobj'] = [ori2mis[io] for io in sample['ifobj'] if ori2mis[io] is not None]

    if args.output:
        for (idx, annotation) in enumerate(samples):
            print(idx, annotation)

    print("Final:", len(samples), 'action phrases extracted')

    path_to_sav = utils.path.extracted(args.dir_extracted, args.datasetid)
    log.info("Saving extracted infos to %s" % path_to_sav)
    with open(path_to_sav, 'wb') as f:
        pickle.dump(samples, f)

if __name__ == '__main__':
    main()
