import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.figure(figsize=(10,10))
plt.axis('off')

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
tx = 0.8
ln_func("Flowchart", tx-0.03, ty, cz, 'i')
ty -= 0.03
ln_func('<if', tx+0.0, ty, cz)
ln_func('$\\it{textspan\_a}$', tx+0.03, ty, cy)
ln_func('>', tx+0.13, ty, cz)
plt.gca().add_patch(patches.FancyArrowPatch((tx+0.13, ty-0.015), (tx+0.1, ty-0.075), connectionstyle="arc3,rad=-.7", color=cz, arrowstyle="Simple,head_width=3,head_length=6"))
ln_func('else', tx+0.12, ty-0.0325, cz, 'b')
plt.gca().add_patch(patches.FancyArrowPatch((tx, ty-0.015), (tx, ty-0.135), connectionstyle="arc3,rad=.7", color=cz, arrowstyle="Simple,head_width=3,head_length=6"))
ln_func('then', tx-0.0625, ty-0.065, cz, 'b')
ty-= 0.06
ln_func('$\\it{textspan\_b}$', tx+0.0, ty, ci)
ty -= 0.06
ln_func('$\\it{textspan\_c}$', tx+0.0, ty, ct)

ty = 1
tx = 0.0
ln_func("TRANSCRIPT", tx-0.03, ty, cz, 'i')
ty -= 0.05
ln_func("transcript_a", tx+0.00, ty, ci)
ty -= 0.05
ln_func("transcript_b", tx+0.00, ty, ct)
ty -= 0.05
ln_func("transcript_c", tx+0.00, ty, ct)

plt.text(0.16, 0.9, "Extract", color='w', ha="center", va="center", rotation=0, size=10, bbox={'boxstyle':"rarrow", 'fc':'dodgerblue', 'ec':'dodgerblue'})
plt.text(0.38, 0.9, "Classify", color='w', ha="center", va="center", rotation=0, size=10, bbox={'boxstyle':"rarrow", 'fc':'dodgerblue', 'ec':'dodgerblue'})
plt.text(0.665, 0.9, "Assemble", color='w', ha="center", va="center", rotation=0, size=10, bbox={'boxstyle':"rarrow", 'fc':'dodgerblue', 'ec':'dodgerblue'})

ty = 1.0
tx = 0.22
ln_func("Text Spans", tx-0.03, ty, cz, 'i')
ty -= 0.05
ln_func("$\\it{textspan\_a}$", tx+0, ty, cy)
ty -= 0.05
ln_func("$\\it{textspan\_b}$", tx+0, ty, ci)
ty -= 0.05
ln_func("$\\it{textspan\_c}$", tx+0, ty, ct)

ty = 1
tx = 0.45
ln_func("Relations", tx-0.03, ty, cz, 'i')
ty -= 0.03
ln_func("$\\it{textspan\_a}$", tx+0.00, ty, cy)
ln_func("$\\it{textspan\_b}$", tx+0.0, ty-0.02, ci)
ln_func('<next>', tx+0.11, ty-0.01, cz, 'b')
ty -= 0.06
ln_func("$\\it{textspan\_a}$", tx+0.00, ty, cy)
ln_func("$\\it{textspan\_c}$", tx+0.0, ty-0.02, ct)
ln_func('<if>', tx+0.11, ty-0.01, cz, 'b')
ty -= 0.06
ln_func("$\\it{textspan\_b}$", tx+0.00, ty, ci)
ln_func("$\\it{textspan\_c}$", tx+0.0, ty-0.02, ct)
ln_func('<none>', tx+0.11, ty-0.01, cz, 'b')

plt.savefig('figurepredict', dpi=1500)
#plt.show()