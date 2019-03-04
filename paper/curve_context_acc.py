import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

cz2 = (0.7, 0.7, 0.7)
cz = (0.3, 0.3, 0.3)
cy = (0.7, 0.4, 0.12)
ci = (0.1, 0.3, 0.5)
ct = (0.7, 0.2, 0.1)

ax = plt.figure(figsize=(5,5)).gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.grid(True)
ax.set_ylim([35,85])
plt.yticks(list(range(40,90,10)),[str(i) for i in range(40,90,10)])

ax.set_title('Test')

ax.set_xlabel('Context Level $K$')
ax.set_ylabel('Micro F$_1$ Score (%)')

y=[62.1,58.5,70.1,68.4]
x=[0,1,2,3]
bt, = ax.plot(x,y, '--', label='BERT Test',  marker='^')

y=[54.0,64.0,72.2,66.9]
x=[0,1,2,3]
cat, = ax.plot(x,y, '-.', label='C. Attn. Test', marker='^')

y=[69.3, 66.4, 72.7, 68.8]
x=[0,1,2,3]
cet, = ax.plot(x,y, '-.', label='C. Emb. Test', marker='^')

y=[54.6,62.1,69.0,69.9]
x=[0,1,2,3]
mat, = ax.plot(x,y, '-', label='Mask$_{AVG}$ Test', marker='o')

y=[62.0,64.0,72.6,71.1]
x=[0,1,2,3]
mmt, = ax.plot(x,y, '-', label='Mask$_{MAX}$ Test', marker='o')

y=[49.2, 58.3]
x=[0,3]
ht, = ax.plot(x,y, ':', label='HBMP Test', marker='s')

plt.legend(handles=[bt, cat, cet, mat, mmt, ht])

#plt.show()
plt.savefig('curvetest.png', dpi=1500)