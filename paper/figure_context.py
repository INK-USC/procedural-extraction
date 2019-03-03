import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.axis('off')

text = [
    "It goes under preparation.",
    "We need to",
    "$\\bf{take\ consent}$",
    "from the family.",
    "It doesn’t have anything to ...",
]

text2 = [
    'It doesn’t have anything to ...',
    'You do that, then you', 
    '$\\bf{prep\ the\ patient.}$',
    'Some people will probably tell you...',
    'I don’t do any of that.'
]

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
tx = 0
ln_func('$\\bf{t}_{c}$', tx - 0.06, ty-0.03, cz, 'i')
ln_func('=        (', tx-0.01, ty-0.03, cz)
ln_func(text[0], tx, ty, ci)
ln_func('$\\bf{t}_{-2}$', tx + 0.11, ty-0.03, ci, 'i')
tx += 0.24
ln_func(',', tx-0.01, ty-0.03, cz)
ln_func(text[1], tx, ty, cy)
ln_func('$\\bf{t}_{-1}$', tx + 0.04, ty-0.03, cy, 'i')
tx += 0.11
ln_func(',', tx-0.01, ty-0.03, cz)
ln_func(text[2], tx, ty, ct)
ln_func('$\\bf{t}$', tx + 0.06, ty-0.03, ct, 'i')
tx += 0.135
ln_func(',', tx-0.01, ty-0.03, cz)
ln_func(text[3], tx, ty, cy)
ln_func('$\\bf{t}_{+1}$', tx + 0.06, ty-0.03, cy, 'i')
tx += 0.15
ln_func(',', tx-0.01, ty-0.03, cz)
ln_func(text[4], tx, ty, ci)
ln_func('$\\bf{t}_{+2}$', tx + 0.1, ty-0.03, ci, 'i')
ln_func('                )', tx + 0.1, ty-0.03, cz)

ty = 0.92
tx = -0.1
ln_func('$\\bf{t}_{c}$', tx +0.04, ty-0.03, cz, 'i')
ln_func('=        (', tx+0.09, ty-0.03, cz)
ln_func(text2[0], tx, ty, ci)
ln_func('$\\bf{t}_{-2}$', tx + 0.19, ty-0.03, ci, 'i')
tx += 0.27
ln_func(',', tx-0.01, ty-0.03, cz)
ln_func(text2[1], tx, ty, cy)
ln_func('$\\bf{t}_{-1}$', tx + 0.1, ty-0.03, cy, 'i')
tx += 0.2
ln_func(',', tx-0.01, ty-0.03, cz)
ln_func(text2[2], tx, ty, ct)
ln_func('$\\bf{t}$', tx + 0.07, ty-0.03, ct, 'i')
tx += 0.175
ln_func(',', tx-0.01, ty-0.03, cz)
ln_func(text2[3], tx, ty, cy)
ln_func('$\\bf{t}_{+1}$', tx + 0.13, ty-0.03, cy, 'i')
tx += 0.33
ln_func(',', tx-0.01, ty-0.03, cz)
ln_func(text2[4], tx, ty, ci)
ln_func('$\\bf{t}_{+2}$', tx + 0.1, ty-0.03, ci, 'i')
ln_func('                )', tx + 0.1, ty-0.03, cz)

#plt.savefig('figure1', dpi=1500)
plt.show()