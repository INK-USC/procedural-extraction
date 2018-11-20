import re
import json
import pickle
import os

from tqdm import tqdm
from pycorenlp import StanfordCoreNLP

from . import nlp_server

version = "1.0"

# File name
file_path = "data/01.src.txt"

# Display options
IF_DISP_PREFIX = False
IF_DISP_TQDM = False
IF_DISP_VB_UNMATCH = False
IF_DISP_IF_UNMATCH = False
IF_DISP_BAN = False
IF_DISP_ALL_SEN = False
IF_VERB_ONLY = True

# Character filter
character_patterns = [
    '^Craig:.*',
    '^Cestero:.*',
    ]
def get_name(c_idx):
    return character_patterns[c_idx][1:-3]

# Verb Extraction
VBpattern = '([pos:/NN.*|VBG/]*[pos:/VB|VBP|VBG/][!pos:/VB|VBP|VBZ|,|./]*[pos:/NN.*|VBG/]+)'
ban_words = [
    {'name': 'linking verbs', 'list' :['is ','are ',"'m ","'s "' am ','be ',"'re ",'was ','were ',]},
    {'name': 'other verbs',   'list' :['know ','think ','worry ',]},
    {'name': 'other norms',   'list' :['contradiction ', 'problem ']},
    #{'name': 'punct', 'list' :[',', '.']},
]
ban_pre_word = [
    'to',
]
def contain_ban_word(text):
    skip = False
    ban_type = ""
    ban_word = ""
    low = text.lower()
    for bw_kv in ban_words:
        for banword in bw_kv['list']:
            if banword in low:
                skip, ban_type, ban_word = True, bw_kv['name'], banword
                break
    return skip, ban_type, ban_word


def get_ann(line):
    return nlp_server.annotate(line, properties={
        "timeout": "10000",
        'annotators': 'tokenize,ssplit,depparse',
        'outputFormat': 'json'
    })

def get_reg(line):
    return nlp_server.tokensregex(line, VBpattern, filter=False)

def get_dep(ann, dep, governor, start=0):
    for s_idx, s in enumerate(ann['sentences']):
        for dep_id in range(start, len(s['enhancedPlusPlusDependencies'])):
            tdep = s['enhancedPlusPlusDependencies'][dep_id]
            if dep in tdep['dep'] and (governor == tdep['governor'] or governor == None):
                return tdep['governor'], tdep['dependent'], dep_id+1
    return None, None, 0

def get_token(ann):
    for s_idx, s in enumerate(ann['sentences']):
        return ['ROOT'] + [token['word'] for token in s['tokens']]

def get_token_str(token, start, end):
    output = ""
    if start > end:
        return output
    for idx in range(start, end):
        output += token[idx] + ' '
    return output

def get_next_punct(token, start):
    for idx in range(start, len(token)):
        tok = token[idx]
        if tok == ',' or tok == '.':
            return idx
    return len(token)

def filter_verb(sen, ann, reg):
    token = get_token(ann)
    ann = reg
    if not isinstance(ann, dict):
        return [(False, "No matching", IF_DISP_VB_UNMATCH)]
    output = []
    for l in ann['sentences']:
        for i in range(l['length']):
            text = l[str(i)]['1']['text']
            begin_pos = l[str(i)]['1']['begin']
            end_pos = l[str(i)]['1']['end']
            skip, ban_type, ban_word = contain_ban_word(text)
            if skip:
                continue
            elif begin_pos-1 >= 0 and token[begin_pos-1].lower() in ban_pre_word:
                continue
            else:
                output.append((text, begin_pos, end_pos))
                
    return output

def filter_if(sen, ann, reg):
    output = []    
    token = get_token(ann)
    start = 0
    while(1):
        if_gov, if_dep, start = get_dep(ann, 'advcl:if', None, start)
        if if_gov is None:
            return output
        next_punct = get_next_punct(token, if_dep)
        punct_pos = min(next_punct, if_gov-1) if if_gov > if_dep else next_punct
        _, mark_dep, _ = get_dep(ann, 'mark', if_dep)
        if if_gov and punct_pos and mark_dep and punct_pos > mark_dep:
            output.append(("{}(pos{}) <condition of> {}".format(token[if_gov], if_gov, get_token_str(token, mark_dep, punct_pos))))
    return output

# split sentences
def split_s(line):
    ann = nlp_server.annotate(line, properties={
        "timeout": "10000",
        'annotators': 'ssplit',
        'outputFormat': 'json'
    })

    output = []
    for s in ann['sentences']:
        idx_start = s['tokens'][0]['characterOffsetBegin']
        idx_end = s['tokens'][-1]['characterOffsetEnd']
        output.append(line[idx_start:idx_end])
    return output

def main():
    filter_sen("You are a good guy.")


memory = {}

def filter_sen(line):
    line = line.strip()
    line = line.replace('’','\'').replace('“','\'').replace('”','\'').replace('‘','\'').replace('—','-').replace('…','...').replace('––','-')
    if line in memory:
        return memory[line]
    outputs = []
    sens = split_s(line)
    was = len(outputs)
    for s_idx, sen in enumerate(sens):
        ann = get_ann(sen)
        reg = get_reg(sen)
        outputs += filter_verb(sen, ann, reg)
    memory[line] = outputs
    return outputs

def load_aps(path):
    global memory
    if os.path.isfile(path):
        print('loading aps from {}'.format(path))
        with open(path, 'rb') as f:
            memory = pickle.load(f)
    else:
        print('no exists aps at {}'.format(path))

def save_aps(path):
    with open(path, 'wb') as f:
        pickle.dump(memory, f)


if __name__ == '__main__':
    main()