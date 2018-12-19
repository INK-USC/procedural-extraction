import pickle
import logging

log = logging.getLogger(__name__)

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
    dataset = {
        'next': [],
        'if': [],
        'none': []
    }
    for (idx, sample) in enumerate(samples):
        if sample['src_matched'] is None:
            continue
        text1 = ' '.join(sample['src_matched']['span'])
        for (idx2, sample2) in enumerate(samples):
            if sample2['src_matched'] is None or idx == idx2:
                continue
            text2 = ' '.join(sample2['src_matched']['span'])
            if idx2 == sample['next_id']:
                dataset['next'].append((text1, text2, 'next'))
            elif sample['iftype'] == 'THEN' and idx2 in sample['ifobj']:
                dataset['if'].append((text1, text2, 'if'))
            else:
                dataset['none'].append((text1, text2, 'none'))
        
    log.info('Saving relation dataset to {}'.format(path))
    with open(path, 'w') as f:
        for (k, v) in dataset.items():
            for triplet in v:
                f.write("%s\t%s\t%s\n" % triplet)
