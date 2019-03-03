import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.figure(figsize=(10,10))
plt.axis('off')

cz2 = (0.7, 0.7, 0.7)
cz = (0.3, 0.3, 0.3)
cy = (0.7, 0.4, 0.12)
ci = (0.1, 0.3, 0.5)
ct = (0.7, 0.2, 0.1)

def ln_func(text, x, y, color, mode=None): 
    if mode is None:
        color_font = b_color = color
        alpha_bg = 0
    elif mode == 'i':
        color_font = 'white'
        b_color = color
        alpha_bg = 1
    elif mode == 'b':
        color_font = b_color = color
        color = 'white'
        alpha_bg = 1
    plt.text(x, y, text, ha="left", va="top", color=color_font, bbox=dict(boxstyle="round", alpha=alpha_bg, ec=b_color, fc=color))

def ln_func2(text, x, y, color, mode=None): 
    if mode is None:
        color_font = b_color = color
        alpha_bg = 0
    elif mode == 'i':
        color_font = 'white'
        b_color = color
        alpha_bg = 1
    elif mode == 'b':
        color_font = b_color = color
        color = 'white'
        alpha_bg = 1
    plt.text(x, y, text, ha="center", va="top", fontsize=16, color=color_font, bbox=dict(boxstyle="round", alpha=alpha_bg, ec=b_color, fc=color))

ty = 0.8
tx = 0.2
plt.arrow(tx, ty, 0.48-tx, 0.85-ty, color=ci)
ln_func('$\\bf{h}_{[cls]}$', tx, ty, ci, 'i')
tx += 0.06
ln_func('...', tx, ty, cz)
tx += 0.03
ln_func('$\\bf{h}_{-1}$', tx, ty, ct, 'b')
tx += 0.05
plt.arrow(tx, ty, 0.48-tx, 0.85-ty, color=ct)
ln_func('$\\bf{h}$', tx, ty, ct, 'i')
tx += 0.03
plt.arrow(tx, ty, 0.48-tx, 0.85-ty, color=ct)
ln_func('$\\bf{h}$', tx, ty, ct, 'i')
tx += 0.03
plt.arrow(tx, ty, 0.48-tx, 0.85-ty, color=ct)
ln_func('$\\bf{h}$', tx, ty, ct, 'i')
tx += 0.03
ln_func('$\\bf{h}_{+1}$', tx, ty, ct, 'b')
tx += 0.05
ln_func('        Pooling        ', tx-0.08, ty+0.07, cz, 'b')
plt.arrow(tx, ty+0.07, 0, 0.021, color=cz)
ln_func('      Class Label     ', tx-0.08, ty+0.12, cz, 'i')
ln_func('...', tx, ty, cz)
tx += 0.035
ln_func('$\\bf{h\'}_{-1}$', tx, ty, cy, 'b')
tx += 0.06
plt.arrow(tx, ty, 0.48-tx, 0.85-ty, color=cy)
ln_func('$\\bf{h\'}$', tx, ty, cy, 'i')
tx += 0.035
plt.arrow(tx, ty, 0.48-tx, 0.85-ty, color=cy)
ln_func('$\\bf{h\'}$', tx, ty, cy, 'i')
tx += 0.035
plt.arrow(tx, ty, 0.48-tx, 0.85-ty, color=cy)
ln_func('$\\bf{h\'}$', tx, ty, cy, 'i')
tx += 0.035
ln_func('$\\bf{h\'}_{+1}$', tx, ty, cy, 'b')
tx += 0.055
ln_func('...', tx, ty, cz)

plt.text(0.27,0.70,'Encoder Model', fontsize=30, color=cz2)
rect = patches.Rectangle((0.175,0.65),0.6,0.17,linewidth=2,edgecolor=cz2,facecolor='none')

# Add the patch to the Axes
plt.gca().add_patch(rect)

#plt.show()
plt.savefig('figuremask', dpi=1500)