import glob, re, numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

# Define kernel
window_size = 128
kernel = 1 - np.cos(np.linspace(0, 2*np.pi, window_size))
kernel /= np.sum(kernel)

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
task_idx = {'3': 0, '4': 1, '7': 2}

for f in glob.glob("losslog_*_t*_test_r*.txt"):
    c, t = re.search(r'_(g|hg)_t(3|4|7)_', f).groups()
    x, y = np.loadtxt(f, usecols=(0, 1)).T
    
    # Apply sliding-window filter and align x-axis
    y = np.convolve(y, kernel, mode='same')
    
    axs[task_idx[t]].plot(x, y, color='r' if c == 'g' else 'b')

for t, i in task_idx.items():
    axs[i].set_title(f"Task {t}")
    axs[i].set_xlabel("Gradient Step")
    axs[i].set_yscale('log', base=np.e)

axs[0].set_ylabel("Loss")

# Force ticks to integer powers of e and format nicely as e^x
axs[0].yaxis.set_major_locator(LogLocator(base=np.e, subs=(1.0,)))
axs[0].yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"$e^{{{int(np.round(np.log(val)))}}}$"))

axs[0].plot([], [], 'r', label='g')
axs[0].plot([], [], 'b', label='hg')
axs[0].legend()

plt.tight_layout()
plt.show()
