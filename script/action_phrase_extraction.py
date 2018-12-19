"""
Extract action phrases to pkls
"""

import pickle
import argparse
import os.path

from pattern_extraction.corenlp import filter_sen
from source_processor import SourceTranscriptProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('datasetid', metavar='N', type=int,
                        help='Datasets to process')
    args = parser.parse_args()
    ds_idx = args.datasetid

    path_to_saving_dir = 'pkls/{:02d}.action_phrases.pkl'.format(ds_idx)
    if os.path.isfile(path_to_saving_dir):
        print('file {} already exists'.format(path_to_saving_dir))
    else:
        src = SourceTranscriptProcessor('data/{:02d}.src.txt'.format(ds_idx), 'data/{:02d}.src.ref.txt'.format(ds_idx))

        outdict = {}
        for line in src.get_all_lines():
            finish = False
            while not finish:
                try:
                    outputs = filter_sen(line)
                    for output in outputs:
                        outdict[output] = 1
                    finish = True
                except Exception:
                    print('waitting server...')
        outlist = list(outdict.keys())

        with open(path_to_saving_dir, 'wb') as f:
            pickle.dump(outlist, f)