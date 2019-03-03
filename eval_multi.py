import json
import sys
import os
import copy
from statistics import pstdev

jsobjs = []

for i in range(1, 6):
    jsobjs.append(json.load(open(os.path.join('logs',sys.argv[1]+'-'+str(i),'metrics.json'), 'r')))

avgobj = copy.copy(jsobjs[0])

for (k, v) in avgobj.items():
    if k == 'hyperparas':
        continue
    for i in range(5):
        p = jsobjs[i][k]['label1_precision']
        r = jsobjs[i][k]['label1_recall']
        f1 = 2 * p * r / (p + r)
        jsobjs[i][k]['label1_f1'] = f1
        p = jsobjs[i][k]['label2_precision']
        r = jsobjs[i][k]['label2_recall']
        f1 = 2 * p * r / (p + r)
        jsobjs[i][k]['label2_f1'] = f1

avgobj = copy.copy(jsobjs[0])

for (k, v) in avgobj.items():
    if k == 'hyperparas':
        continue
    newv = {}

    for (k2, v2) in v.items():
        vals = [jsobjs[i][k][k2] for i in range(5)]
        newv[k2] = sum(vals) / len(vals)
        newv[k2+'_std'] = pstdev(vals)
    avgobj[k] = newv

sdir = os.path.join('logs',sys.argv[1]+'-avg')
if not os.path.exists(sdir):
        os.makedirs(sdir)
else:
    print('overwriting', sdir)
json.dump(avgobj, open(os.path.join(sdir,'metrics.json'), 'w'), indent=4, sort_keys=True)