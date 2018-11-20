import pickle
import calc_dist
import argparse
import source_transcript_processor
import sys
from  corenlp import filter_sen, load_aps, save_aps

"""
Compare the filtered APs with the parsed TGTs
"""

dist_calculator = calc_dist.CalcDistance()

ds_idx = int(sys.argv[1])
part = sys.argv[2]
if_refresh_aps = sys.argv[3]

with open('data/{:02d}.{}.dataset.pkl'.format(ds_idx, part), 'rb') as f:
    obj = pickle.load(f)

if if_refresh_aps != 'y':
    load_aps('pkls/{:02d}.{}.aps.pkl'.format(ds_idx, part))

def get_refs(annotations):
    if annotations is None:
        return []
    return [' '.join(an['span']) for an in annotations]

etp = 0
ctp = 0
eaps = 0
refs = 0

avgp = 0
avgr = 0
avgf = 0
cavgp = 0
cavgr = 0
cavgf = 0
avgtot = 0

tok_tp = 0
tok_tn = 0
tok_fp = 0
tok_fn = 0
output = []

html = []
lastline = 0
showname = True

for sen in obj:
    system_outs = filter_sen(' '.join(sen['tokens']))
    correctness = [False] * len(system_outs)
    refere_outs = sen['annotations']

    if len(system_outs):
        print(system_outs)

    tok_an = [{'word': tok} for tok in sen['tokens']]

    if refere_outs is not None:
        avgtot += 1

        setp = 0
        sctp = 0
        seaps = 0
        srefs = 0

        if showname:
            html.append(sen['speaker'])
            showname = False

        srefs += len(refere_outs)
        seaps += len(system_outs)
            
        for ref in refere_outs:
            matched_ap = ""
            mm = ""
            span = ' '.join(ref['span'])
            syss = [sys[0] for sys in system_outs]
            argmin_exact = dist_calculator.calc_dist_multi_hyps(span, syss, 'exact')
            if argmin_exact is not None:
                setp += 1
                sctp += 1
                mm = 'Exact'
                matched_ap = syss[argmin_exact]
                correctness[argmin_exact] = True
            else:
                argmin_contain = dist_calculator.calc_dist_multi_hyps(span, syss, 'contain')
                if argmin_contain is not None:
                    sctp += 1
                    mm = 'Contains'
                    matched_ap = syss[argmin_contain]
                    correctness[argmin_contain] = True
            output.append("{}, {}, {}, {}, {}".format(ref['protocol'].replace(',',' '), ' '.join(ref['span']).replace(',',' '), dist_calculator.calc_dist(ref['protocol'], ' '.join(ref['span']), 'embavg'), matched_ap.replace(',',' '), mm))
            
            s = ref['start']
            e = s + ref['K']
            for pos in range(s, e):
                tok_an[pos]['gt'] = 1

        p = setp / seaps if seaps != 0 else 0
        r = setp / srefs
        cp = sctp / seaps if seaps != 0 else 0
        cr = sctp / srefs
        f = 2 * p * r / (p + r) if p + r != 0 else 0
        cf = 2 * cp * cr / (cp + cr) if cp + cr != 0 else 0

        avgp += p
        avgr += r
        cavgp += cp
        cavgr += cr
        avgf += f
        cavgf += cf

        etp += setp
        ctp += sctp
        eaps += seaps
        refs += srefs

        for (idx, sys) in enumerate(system_outs):
            if not correctness[idx]:
                output.append(", , , {}, ".format(sys[0]))
            b = int(sys[1])
            e = int(sys[2])
            for pos in range(b, e):
                tok_an[pos]['sys'] = 1

        src_txt = []
        for tok in tok_an:
            html_tok = tok['word']
            if 'gt' in tok:
                if 'sys' in tok:
                    html_tok = '<span style="color: coral; font-weight: bold">{}</span>'.format(html_tok)
                    tok_tp += 1
                else:
                    html_tok = '<span style="color: red">{}</span>'.format(html_tok)
                    tok_fn += 1
            elif 'sys' in tok:
                html_tok = '<span style="color: orange">{}</span>'.format(html_tok)
                tok_fp += 1
            else:
                tok_tn += 1
            src_txt.append(html_tok)

        src_txt = ' '.join(src_txt)

        html.append(src_txt)
    else:
        if showname:
            html.append('<span style="color: silver">{}</span>'.format(sen['speaker']))
            showname = False
        html.append('<span style="color: silver">{}</span>'.format(' '.join(sen['tokens'])))
    if sen['line'] != lastline:
        html.append('<br>')
        lastline = sen['line']
        showname = True

print("Token Level")
print("True Positive: {}".format(tok_tp))
print("False Positive: {}".format(tok_fp))
print("True Negative: {}".format(tok_tn))
print("False Negative: {}".format(tok_fn))
print("Precision: {:.2f}%".format(tok_tp / (tok_tp + tok_fp)*100))
print("Recall: {:.2f}%".format(tok_tp / (tok_tp + tok_fn)*100))
print("Text Span Level")
print("Exact Precision: {} / {} = {:.2f}%".format(etp, eaps, etp/eaps*100))
print("Exact Recall: {} / {} = {:.2f}%".format(etp, refs, etp/refs*100))
print("Contain Precision: {} / {} = {:.2f}%".format(ctp, eaps, ctp/eaps*100))
print("Contain Recall: {} / {} = {:.2f}%".format(ctp, refs, ctp/refs*100))
print("Sentence Level")
print("Average Exact Precision: {:.2f}%".format(avgp/avgtot*100))
print("Average Exact Recall: {:.2f}%".format(avgr/avgtot*100))
print("Average Exact F1: {:.2f}%".format(avgf/avgtot*100))
print("Average Contain Precision: {:.2f}%".format(cavgp/avgtot*100))
print("Average Contain Recall: {:.2f}%".format(cavgr/avgtot*100))
print("Average Contain F1: {:.2f}%".format(cavgf/avgtot*100))

save_aps('pkls/{:02d}.{}.aps.pkl'.format(ds_idx, part))

with open('out.csv', 'w') as f:
    f.write('\n'.join(output))
with open('out.html', 'w') as f:
    f.write('\n'.join(html))