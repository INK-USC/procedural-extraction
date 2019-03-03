import json
import sys
import os
import copy
from statistics import pstdev
sets = ['test', 'manual']
sname = ['Test', 'Manual Matching']
firsthead = 'Model'
head = ['Accuracy', 'Micro F\\textsubscript{1}', '$<$next$>$ F\\textsubscript{1}', '$<$if$>$ F\\textsubscript{1}']
keys = ['accuracy', 'f1_micro', 'label1_f1', 'label2_f1']
devs = ['accuracy_std', 'f1_micro_std', 'label1_f1_std', 'label2_f1_std']
names = ["Pos. Attn.", "Seg. Emb.", "BERT", 'Mask'] 
showname = names * 2 + names[2:]
idx = 0

def retrieve_head():
    el = [firsthead]
    ele = ['']
    for (idx, n) in enumerate(sname):
        el.append("\\multicolumn{4}{c%s}{%s}" % ('|' if not idx else '', n))
        for h in head:
            ele.append(h)
    return ' & '.join(el), ' & '.join(ele)

def retrieve_result(name):
    global idx
    avgobj = json.load(open(os.path.join('logs',name+'-avg','metrics.json'), 'r'))
    sh = showname[idx]
    idx += 1
    row_ele = [sh]
    for s in sets:
        for k, d in zip(keys, devs):
            row_ele.append("%.1f \\textpm %.1f" % (avgobj[s][k]*100, avgobj[s][d]*100))
    return ' & '.join(row_ele)

sts = "Sampling portion = $1 : 1 : 1$ (0.4k samples)"

rows = '\\\\\n'.join([*retrieve_head()] + ['']) + '\\midrule\n\\multicolumn{9}{c}{%s6 : 3 : 1}\\\\\n\\midrule\n' % sts + '\\\\\n'.join([retrieve_result(name) for name in [
    "postattn631",
    "segemb631",
    "none631",
    "maskavg631"
    ]]+['']) + '\\midrule\n' + '\\multicolumn{9}{c}{%s4 : 2 : 1}\\\\\n\\midrule\n' % sts + '\\\\\n'.join([retrieve_result(name) for name in [
    "postattn421",
    "segemb421",
    "none421",
    "maskavg421"
    ]]+['']) + '\\midrule\n' + '\\multicolumn{9}{c}{%s1 : 1 : 1}\\\\\n\\midrule\n' % sts + '\\\\\n'.join([retrieve_result(name) for name in [
    'none111',
    'maskavg111'
    ]] + [''])
print(rows)