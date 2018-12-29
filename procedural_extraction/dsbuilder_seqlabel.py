import utils
import argparse
import logging

from procedural_extraction.source_processor import SourceProcessor
from .dsbuilder import register_dsbuilder

log = logging.getLogger(__name__)

register_dsbuilder('seqlabel')
def builder_seqlabel_dataset(parser: argparse.ArgumentParser):
    # TODO finish adapting code to multi-dataset
    def _method(sample_sets):
        args = parser.parse_args()
        path_src = utils.path.src(args.dir_data, args.datasetid)
        path_src_ref = utils.path.src_ref(args.dir_data, args.datasetid)
        log.info("Loading source file %s" % path_src)
        src = SourceProcessor(path_src, path_src_ref, args.src_retok)

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

        dataset = list()
        for (idx, sen) in enumerate(src.src_sens):
            a = None
            if idx in annotations:
                a = annotations[idx]
            dataset.append({
                'id': idx,
                'line': src.sen2src[idx],
                'tokens': sen,
                'annotations': a,
                'speaker': src.src_sens_speakers[idx]
            })

        log.info('Saving sequence labeling dataset to {}'.format(path_src))

    def create_iobes(ds):
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

        with open('data/{:02d}.{}.iobes'.format(ds_idx, suffix), 'w') as f:
            f.write('\n'.join(output_io))
    
    return _method