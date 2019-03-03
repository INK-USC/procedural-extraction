import json
import sys
import os
import copy
from statistics import pstdev

keys = ['accuracy', 'f1_micro', 'label1_precision', ]
devs = ['accuracy_std', 'f1_micro_std', 'label1_precision_std', ]

def retrieve_result(name):
        avgobj = json.load(open(os.path.join('logs',name+'-avg','metrics.json'), 'r'))

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

for name in [
    "none421",
    "none421_n1",
    "none421_n0",
    "none421_nc",
    "maskavg421",
    "maskavg421_n1",
    "maskavg421_n0",
    "maskavg421_nc"
]