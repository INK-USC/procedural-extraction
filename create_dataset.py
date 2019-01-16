import pickle
import logging
import argparse
import os
import utils

import procedural_extraction

log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='process extracted infos to dataset in different shapes.')
    parser.add_argument('type', choices=procedural_extraction.get_builder_names(), help='dataset type to build')
    # saving
    parser.add_argument('--dataset', default='1,2,3,4,6', help='specify id of dataset to use, or use all 5 available datasets by default')
    parser.add_argument('--dir_data', default='data', help='specify dir holding source data')
    parser.add_argument('--dir_extracted', default='extracted', help='specify dir save extracted data')
    # misc
    parser.add_argument("--verbosity", help="logging verbosity", default="INFO")

    args, extra = parser.parse_known_args()
    logging.basicConfig(level=getattr(logging, args.verbosity), handlers=[logging.StreamHandler()])

    sample_sets = list()
    for dsid in map(int, args.dataset.split(',')):
        path_to_sav = utils.path.extracted(args.dir_extracted, dsid)
        log.info("Loading infos from %s" % path_to_sav)
        with open(path_to_sav, 'rb') as f:
            sample_sets.append((dsid, pickle.load(f)))

    builder = procedural_extraction.get_builder(args.type, parser)
    log.info("Creating datasets")
    builder(sample_sets)

if __name__ == "__main__":
    main()