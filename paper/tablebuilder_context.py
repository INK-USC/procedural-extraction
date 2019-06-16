import json
import sys
import os
import copy
from statistics import pstdev
sets = ['test', 'manual']
sname = ['Test', 'Manual Matching']
firsthead = 'Setting'
head = ['Accuracy', 'Micro F\\textsubscript{1}', '$<$next$>$ F\\textsubscript{1}', '$<$if$>$ F\\textsubscript{1}']
keys = ['accuracy', 'f1_micro', 'label1_f1', 'label2_f1']
devs = ['accuracy_std', 'f1_micro_std', 'label1_f1_std', 'label2_f1_std']
showname = ['BERT \\textsubscript{K=%d}' % i for i in range(3, -1, -1)] + ['C. Attn. \\textsubscript{K=%d}' % i for i in range(3, -1, -1)] + ['C. Emb. \\textsubscript{K=%d}' % i for i in range(3, -1, -1)] + ['Mask \\textsubscript{AVG} \\textsubscript{K=%d}' % i for i in range(3, -1, -1)] + ['Mask \\textsubscript{MAX} \\textsubscript{K=%d}' % i for i in range(3, -1, -1)]
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


rows = '\\\\\n'.join([*retrieve_head()] + ['']) + '\\midrule\n' + '\\\\\n'.join([retrieve_result(name) for name in [
"none421",
"none421_n1",
"none421_n0",
"none421_nc",
    ]]+['']) + '\\midrule\n'+ '\\\\\n'.join([retrieve_result(name) for name in [
"postattn421",
"postattn421_n1",
"postattn421_n0",
"postattn421_nc",
    ]]+['']) + '\\midrule\n'+ '\\\\\n'.join([retrieve_result(name) for name in [
"segemb421",
"segemb421_n1",
"segemb421_n0",
"segemb421_nc",
    ]]+['']) + '\\midrule\n' + '\\\\\n'.join([retrieve_result(name) for name in [
"maskavg421",
"maskavg421_n1",
"maskavg421_n0",
"maskavg421_nc",
    ]]+['']) + '\\midrule\n'+ '\\\\\n'.join([retrieve_result(name) for name in [
"maskmax421",
"maskmax421_n1",
"maskmax421_n0",
"maskmax421_nc",
    ]] + [''])
print(rows)