import argparse
import pickle
import sys

if __name__ == '__main__':
    ds_idx = int(sys.argv[1])
    suffix = sys.argv[2]

    with open('data/{:02d}.{}.dataset.pkl'.format(ds_idx, suffix), 'rb') as f:
        obj = pickle.load(f)

    output_io = list()
    for sen in obj:
        toks = sen['tokens']
        refs = sen['annotations']
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