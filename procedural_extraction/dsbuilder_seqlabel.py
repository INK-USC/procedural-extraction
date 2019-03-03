import utils
import argparse
import logging
import numpy as np

from procedural_extraction.source_processor import SourceProcessor
from .dsbuilder import register_dsbuilder

log = logging.getLogger(__name__)

@register_dsbuilder('seqlabel')
def builder_seqlabel_dataset(parser: argparse.ArgumentParser):
    parser.add_argument('--path', default='dataset/seqlab', metavar='path_to_dir', help='dir to save the dataset')
    parser.add_argument('--seed', default=42)

    args = parser.parse_args()


    def _method(sample_sets):
        args = parser.parse_args()
        dataset = list()
        for (dsid, samples) in sample_sets:
            path_src = utils.path.src(args.dir_data, dsid)
            path_src_ref = utils.path.src_ref(args.dir_data, dsid)
            log.info("Loading source file %s" % path_src)
            src = SourceProcessor(path_src, path_src_ref, False)

            annotations = dict()
            def add_matched_ngram(oris, protocol_text):
                id = oris['src_sens_id']
                oris['protocol'] = protocol_text
                if id not in annotations:
                    annotations[id] = list()
                if oris['span'] not in [a['span'] for a in annotations[id]]:
                    annotations[id].append(oris)
        
            for sample in samples:
                matched = sample['src_matched']
                text = sample['text']
                if matched is not None:
                    add_matched_ngram(matched, text)

            for (idx, sen) in enumerate(src.src_sens):
                a = None
                if idx in annotations:
                    a = annotations[idx]
                dataset.append({
                    'id': str(dsid) + '-' + str(idx),
                    'line': src.sen2src[idx],
                    'tokens': sen,
                    'annotations': a,
                    'speaker': src.src_sens_speakers[idx]
                })
        np.random.seed(args.seed)
        np.random.shuffle(dataset)
        tot = len(dataset)
        part = tot // 8
        create_iobes(dataset[:part*6], 'train')
        create_iobes(dataset[part*6:part*7], 'dev')
        create_iobes(dataset[part*7:], 'test')
        create_iobes(dataset, 'nonsplit')

        print(tot)

    def create_iobes(ds, split):
        output_io = list()
        for sen in ds:
            toks = sen['tokens']
            refs = sen['annotations']
            # negative samples?
            if refs is None:
                continue

            anns = ['O'] * len(toks)
            for ref in refs:
                s = ref['start']
                e = s + ref['K']
                if s == e - 1:
                    anns[s] = 'S-VB'
                else:
                    anns[s] = 'B-VB'
                    for pos in range(s + 1, e-1):
                        anns[pos] = 'I-VB'
                    anns[e-1] = 'E-VB'

            for (tok, ann) in zip(toks, anns):
                io_line = tok + '\t' + ann
                output_io.append(io_line)

            output_io.append('')

        path = '%s/%s.iobes' % (args.path, split)
        with open(path, 'w') as f:
            f.write('\n'.join(output_io))
        log.info('Saving sequence labeling dataset to %s' % path)
    
    return _method