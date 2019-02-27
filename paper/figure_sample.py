import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.axis('off')

i_text = "It goes under preparation."
ie_text = "We need to $\\bf{take\ consent}$ from the\n family.  It doesn't have anything to do"
ie_2 = "with the actual procedure, but medical\n legal stuff.  You do that, then you"
ie_3 = "$\\bf{prep\ the\ patient}$. Some people will \n probably tell you they put a roll ..."

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
ln_func("Interviewer", 0.0, ty, ci, 'b')
ln_func('117', 0.13, ty, ci, 'i')
ln_func(i_text, 0.18, ty, ci)
ty -= 0.04
ln_func("Interviewee", 0.0, ty, ct, 'b')
ln_func('118', 0.13, ty, ct, 'i')
ln_func(ie_text, 0.18, ty + 0.01, ct)
ty -= 0.04
ln_func('119', 0.13, ty, ct, 'i') 
ln_func(ie_2, 0.18, ty + 0.01, ct)
ty -= 0.04
ln_func('120', 0.13, ty, ct, 'i')
ln_func(ie_3, 0.18, ty + 0.01, ct)

ty -= 0.04
ln_func(" PROTOCOL ", -0.03, ty, cz, 'i')
ty -= 0.04
ln_func('Task A.3   ', 0.018, ty, cy, 'b')
ln_func('Prepare patient (120) and self (133)', 0.13, ty, cy)
ln_func('120', 0.2825, ty, cy, 'i')
ln_func('133', 0.4125, ty, cy, 'i')

ty-= 0.04
ln_func('Step A.3.1', 0.018, ty, ci, 'b')
ln_func('Obtain consent  118 ', 0.13, ty, ci)
ln_func('118', 0.2825, ty, ci, 'i')

ty -= 0.04
ln_func('Step A.3.2', 0.018, ty, ct, 'b')
ln_func('Identify and locate equipment', 0.13, ty, ct)
ln_func('308', 0.4125, ty, ct, 'i')

plt.savefig('figure1', dpi=1500)