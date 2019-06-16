import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

cz2 = (0.7, 0.7, 0.7)
cz = (0.3, 0.3, 0.3)
cy = (0.7, 0.4, 0.12)
ci = (0.1, 0.3, 0.5)
ct = (0.7, 0.2, 0.1)

ax = plt.figure(figsize=(5,4)).gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.grid(True)
ax.set_ylim([35,85])
plt.yticks(list(range(40,90,10)),[str(i) for i in range(40,90,10)])

ax.set_title('Manual Matching')

ax.set_xlabel('Context Level $K$')
ax.set_ylabel('Micro F$_1$ Score (%)')

y=[50.7,60.2, 62.2,63.6]
x=[0,1,2,3]
bm, = ax.plot(x,y, '--', label='BERT MM', marker='^')

y=[51.7,62.1,72.7,70.9]
x=[0,1,2,3]
cam, = ax.plot(x,y, '-.', label='C. Attn. MM', marker='^')

y=[52.2, 52.4, 67.4, 71.5]
x=[0,1,2,3]
cem, = ax.plot(x,y, '-.', label='C. Emb. MM', marker='^')

y=[49.9,54.6,73.4,74.4]
x=[0,1,2,3]
mam, = ax.plot(x,y, '-', label='Mask$_{AVG}$ MM', marker='o')

y=[52.7,53.9,81.4,78.6]
x=[0,1,2,3]
mmm, = ax.plot(x,y, '-', label='Mask$_{MAX}$ MM', marker='o')

y=[39.6, 54.5, 63.3, 64.8]
x=[0,1,2,3]
hm, = ax.plot(x,y, ':', label='HBMP MM', marker='s')

y=[2, 54.5, 63.3, 64.8]
x=[0,1,2,3]
pm, = ax.plot(x,y, ':', label='PCNN MM', marker='s')

plt.legend(handles=[bm, cam, cem, mam, mmm, hm])

#plt.show()
plt.savefig('curvemanual.png', dpi=1500)