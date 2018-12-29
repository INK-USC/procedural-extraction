import pickle
import logging
import argparse
import os
import utils

import procedural_extraction

log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='process extracted infos to dataset in different shapes.')
    parser.add_argument('method', help='method of fuzzy matching')
    parser.add_argument('type', help='dataset type to build')
    # saving
    parser.add_argument('--dir_extracted', default='extracted', help='specify dir save extracted data')
    parser.add_argument('--dir_seqlabel', default='dataset/seqlabel', help='specify dir save seq label dataset')
    parser.add_argument('--dir_relation', default='dataset/relation', help='specify dir save phrase pair relation dataset')

    args, extra = parser.parse_known_args()
    logging.basicConfig(level=getattr(logging, args.verbosity), handlers=[logging.StreamHandler()])

    sample_sets = list()
    for dsid in [1,2,3,4,6]:
        path_to_sav = utils.path.extracted(args.dir_extracted, dsid)
        log.info("Loading infos from %s" % path_to_sav)
        with open(path_to_sav, 'rb') as f:
            sample_sets.append(pickle.load(f))

    builder = procedural_extraction.get_builder(args.type, parser)
    log.info("Creating datasets")
    builder(sample_sets)

if __name__ == "__main__":
    main()