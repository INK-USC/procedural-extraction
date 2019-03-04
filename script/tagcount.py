ttp = 0
ttot = 0
mtp = 0
mltot = 0
mptot = 0
beg = False

with open('output.log', 'r') as f:
    for line in f:
        line = line.strip()
        if not len(line):
            continue
        line = line.split('\t')
        if line[1] == line[2]:
            ttp += 1
        ttot += 1
        if line[1] == line[2] and line[1] == 'B-VB':
            beg = True
        if line[1] == 'E-VB' and line[2] != 'E-VB':
            beg = False
        if line[2] == 'E-VB' and line[1] != 'E-VB':
            beg = False
        if line[1] == line[2] and line[1] == 'E-VB' and beg:
            mtp += 1
        if line[1] == 'B-VB':
            mltot += 1
        if line[2] == 'B-VB':
            mptot += 1

print('tokenlevel acc.:', ttp / ttot)
print('mention level')
p = mtp / mptot
r = mtp / mltot
print('P', p)
print('R', r)
print('F1', (2*p*r)/(p+r))