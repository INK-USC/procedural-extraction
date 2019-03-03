import pickle
import argparse
import os.path
import json
from tqdm import tqdm

from pattern_extraction.corenlp import filter_sen
from procedural_extraction.source_processor import SourceProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('datasetid', metavar='N', type=int,
                        help='Datasets to process')
    args = parser.parse_args()
    ds_idx = args.datasetid

    src = SourceProcessor('data/{:02d}.src.txt'.format(ds_idx), 'data/{:02d}.src.ref.txt'.format(ds_idx))

    outdict = {}
    for line in tqdm(src.get_all_lines()):
        finish = False
        while not finish:
            try:
                outputs = filter_sen(line)
                for output in outputs:
                    outdict[output] = 1
                finish = True
            except Exception:
                print('waitting server...')
    for o in outdict:
        print(o)
    outlist = [o[0] for o in outdict]

    obj = json.load(open('answer.json', 'r'))
    obj = [e['zcandidates'][0] for e in obj if len(e['zcandidates'])]

    tp = 0
    for o in obj:
        if o in outlist:
            tp += 1
    tpfp = len(outlist)
    tpfn = len(obj)
    p = tp / tpfp
    r = tp / tpfn

    print('P', p, '  R', r, '  F1', (2 * p * r)/(p+r))
