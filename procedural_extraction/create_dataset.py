import pickle
import logging as log

def create_seqlabel_dataset(samples, src, path):
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

    log.info('Saving sequence labeling dataset to {}'.format(path))
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)

def create_relation_dataset(samples, path):
    with open(path) as f:
        f.write('\t'.join(["Str1", "Str2", "Label"]) + '\n')
        for sample in samples:
            f.write()
