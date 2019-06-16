import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.axis('off')	
 	
i_text = "Oh, before that, so now you have \nyour wire in, you $\\bf{remove\ your\ needle}$.  "
ie_text = "$\\bf{Leave\ the\ wire\ in\ place}$.  Then,\nactually, before the dilator I'll use my "
ie_2 = "scalpel, and I'll $\\bf{make\ a\ nick\ in\ the\ skin}$\n$\\bf{wide\ enough\ for\ whatever\ catheter}$ "

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

ty = 1
ln_func("TRANSCRIPT", -0.03, ty, cz, 'i')
ty -= 0.04
ln_func("Interviewee", 0.0, ty, ci, 'b')
ln_func('596', 0.13, ty, ci, 'i')
ln_func(i_text, 0.18, ty, ci)
ty -= 0.08
ln_func('597', 0.13, ty, ct, 'i')
ln_func(ie_text, 0.18, ty + 0.01, ct)
ty -= 0.08
ln_func('598', 0.13, ty, ct, 'i') 
ln_func(ie_2, 0.18, ty + 0.01, ct)

ty -= 0.04
ln_func(" PROTOCOL ", -0.03, ty, cz, 'i')
ty -= 0.04
ln_func('Step A.5.1', 0.018, ty, cy, 'b')
ln_func('Remove needle              \nand leave wire in place ', 0.13, ty, cy)
ln_func('162', 0.2825, ty + 0.01, cy, 'i')
ln_func('596', 0.3325, ty + 0.01, cy, 'i')
ln_func('597', 0.3525, ty - 0.025, cy, 'i')

ty-= 0.08
ln_func('Step A.5.2', 0.018, ty, ci, 'b')
ln_func('Make a nick in the skin that is \n wide enough for the catheter', 0.13, ty, ci)
ln_func('598', 0.415, ty-0.015, ci, 'i')

ty -= 0.08
ln_func('Step A.5.3', 0.018, ty, ct, 'b')
ln_func('Pass dilator', 0.13, ty, ct)
ln_func('162', 0.245, ty, ct, 'i')
ln_func('602', 0.295, ty, ct, 'i')

#plt.show()
plt.savefig('figure1', dpi=1500)