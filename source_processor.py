from pattern_extraction.corenlp import nlp_server
import os.path
import pickle
import re
import time

class SourceTranscriptProcessor:
    def __init__(self, path, path_ref):
        with open(path, 'r') as f:
            self.lines = f.readlines()
        with open(path_ref, 'r') as f:
            self.lines_ref = f.readlines()

        """
        src_tokenized[line_number][sentence_number][tokens][token_number]
        """
        self.path = path
        self.src_tokenized = self.load_tokenize(self.lines, path)
        self.src_sens, self.src_sen_map, self.src_sens_speakers, self.sen2src = self.map_src_sentences()
        self.annotations = {}

        self.ref2srcmap, self.ref2senmap = self.build_mapping()
        print(len(self.ref2srcmap))

    def map_src_sentences(self):
        sen_idx = 0
        sens = []
        maps = []
        speakers = []
        sen2src = []
        for (sid, sen) in enumerate(self.src_tokenized):
            amap = {}
            for s in sen:
                for idx in range(s['start'], s['end']+1):
                    amap[idx] = sen_idx
                sens.append(s['tokens'])
                sen2src.append(sid)
                speakers.append(s['speaker'])
                sen_idx += 1
            maps.append(amap)
        return sens, maps, speakers, sen2src

    def load_tokenize(self, lines, path):
        ret = []
        path_to_saving_path = path + '.tok.pkl'
        if os.path.isfile(path_to_saving_path):
            print('file {} already exists'.format(path_to_saving_path))
            with open(path_to_saving_path, 'rb') as f:
                ret = pickle.load(f)
        else:
            for idx, line in enumerate(lines):
                m = re.match("^\D*:\s*", line)
                line = re.sub("^\D*:\s*", '', line)
                toked = self.tokenize(line)
                if m is not None:
                    for sen in toked:
                        sen['speaker'] = m.group()
                else:
                    for sen in toked:
                        sen['speaker'] = '-:\t'
                ret.append(toked)
                print('\r processing line {}'.format(idx), end='')
            print('')
            with open(path_to_saving_path, 'wb') as f:
                pickle.dump(ret, f)

        return ret

    def get_spanning(self, line_num):
        raise NotImplementedError('')

    def tokenize(self, line):
        finish = False
        while not finish:
            try:
                ann = nlp_server.annotate(line, properties={
                    "timeout": "10000",
                    'annotators': 'tokenize,ssplit',
                    'outputFormat': 'json'
                })
                finish = True
            except Exception:
                print('waitting')
                time.sleep(0.5)
        return [
            {
                'tokens': [tok['word'] for tok in sen['tokens']],
                'start': sen['tokens'][0]['characterOffsetBegin'],
                'end': sen['tokens'][-1]['characterOffsetEnd']
            } for sen in ann['sentences']
        ]

    def build_mapping(self):
        ref2srcmap = [-1]
        ref2senmap = [-1]
        start = 0
        for idx_ref, line_ref in enumerate(self.lines_ref):
            line_ref = line_ref.replace('’', '\'').replace('”', '\'').replace("“", '\'')
            line_ref = re.sub("^\D*:\s*", '', line_ref)
            found = False
            for idx, line in enumerate(self.lines[start:]):
                line = line.replace('’', '\'').replace('”', '\'').replace("“", '\'')
                line = re.sub("^\D*:\s*", '', line)
                match_pos = line.find(line_ref.strip())
                if match_pos != -1 and match_pos + len(line_ref.strip()) - 1 >= 0:
                    ref2srcmap.append(start + idx)
                    start_sen = self.src_sen_map[start + idx][match_pos]
                    end_sen = self.src_sen_map[start + idx][match_pos + len(line_ref.strip()) - 1]
                    ref2senmap.append((start_sen, end_sen))
                    found = True
                    start += idx
                    break

            if not found:
                ref2srcmap.append(-1)
                ref2senmap.append(None)
        return ref2srcmap, ref2senmap

    def get_line(self, line_num):
        return self.lines_ref[line_num-1].strip()

    def get_all_lines(self):
        return [re.sub("^\D*:\s*", '', line_ref.replace('’', '\'').replace('”', '\'').replace("“", '\'')) for line_ref in self.lines]

    def get_src_line(self, ref_line_num):
        return self.lines[self.ref2srcmap[ref_line_num]].strip()

    def get_sens_line(self, ref_line_num):
        if self.ref2senmap[ref_line_num] is None:
            return []
        start, end = self.ref2senmap[ref_line_num]
        return [
            {
                'tokens': self.src_sens[id],
                'src_sens_id': id,
                'speaker': self.src_sens_speakers[id]
             } for id in range(start, end + 1)
        ]
    
    def get_ngrams(self, Nmin=2, Nmax=50):
        if self.src_sens is None:
            return []
        new_oris = []
        for (idx, sen) in enumerate(self.src_sens):
            for K in range(Nmin, Nmax + 1):
                for stt in range(0, len(sen) - K + 1):
                    spanning = sen[stt : stt + K]
                    new_oris.append({
                        'span': spanning,
                        'speaker': self.src_sens_speakers[idx],
                        'src_sens_id': idx,
                        'start': stt,
                        'K': K
                    })

        return new_oris

    def get_ngrams_line(self, ref_line_num, Nmin=2, Nmax=50):
        sens = self.get_sens_line(ref_line_num)
        new_oris = []
        for sa in sens:
            sen = sa['tokens']
            for K in range(Nmin, Nmax + 1):
                for stt in range(0, len(sen) - K + 1):
                    spanning = sen[stt : stt + K]
                    new_oris.append({
                        'span': spanning,
                        'speaker': sa['speaker'],
                        'src_sens_id': sa['src_sens_id'],
                        'start': stt,
                        'K': K
                    })
        return new_oris

    def add_matched_ngram(self, oris, protocol_text):
        id = oris['src_sens_id']
        oris['protocol'] = protocol_text
        if id not in self.annotations:
            self.annotations[id] = []
        if oris['span'] not in [a['span'] for a in self.annotations[id]]:
            self.annotations[id].append(oris)

    def dump_dataset(self):
        dataset = []
        for (idx, sen) in enumerate(self.src_sens):
            a = None
            if idx in self.annotations:
                a = self.annotations[idx]
            dataset.append({
                'id': idx,
                'line': self.sen2src[idx],
                'tokens': sen,
                'annotations': a,
                'speaker': self.src_sens_speakers[idx]
            })

        sav_p = self.path+'.dataset.pkl'
        print('saving to {}'.format(sav_p))
        with open(sav_p, 'wb') as f:
            pickle.dump(dataset, f)
            

if __name__ == '__main__':
    test = SourceTranscriptProcessor('data/01.src.txt', 'data/01.src.ref.txt')
    print(test.src_sens[2])


    

