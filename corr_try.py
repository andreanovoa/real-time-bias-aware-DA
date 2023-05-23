


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt



x = np.arange(500) / 100
sig = -2*np.sin(8 * np.pi * x)
sig2 = np.sin(7 * np.pi * x)
# sig2 = np.append(sig[33:], sig[:33])
corr = signal.correlate(sig2, sig)
corr /= max(corr)
lags = signal.correlation_lags(len(sig), len(sig2)//2)
# sig2_corr = np.append(sig2[lags[np.argmax(corr)]:], sig2[:lags[np.argmax(corr)]])

idx = lags[np.argmax(corr)]

def C(y1, y2):
    # return np.sqrt(np.sum((y1 - y2) ** 2) / np.sum(y1** 2))
    return np.sum((y1 * y2)) / np.sqrt(np.sum(y1 ** 2) * np.sum(y2 ** 2))

CS = []
compute_times = np.linspace(0, len(sig2)//2,  len(sig2)//4, dtype=int)
print(compute_times)
for ii in compute_times:
    sig2_corr = np.append(sig2[ii:], sig2[:ii])
    CS.append(C(sig, sig2_corr))



idx_mine = np.argmax(CS)
print(min(CS))
print(max(CS))


fig, (ax_orig, ax_noise, ax_corr, ax_mine) = plt.subplots(4, 1, sharex=False)
ax_orig.set_title('OG')
ax_orig.plot(sig, 'k', lw=2)
ax_orig.plot(sig2, '--')
ax_noise.set_title('Correlated')
ax_noise.plot(sig,  'k', lw=2)

# ax_noise.plot(np.append(sig2[idx:], sig2[:idx]), 'b--')
# ax_noise.plot(np.append(sig2[idx_mine:], sig2[:idx_mine]), 'r--')
ax_noise.plot(sig2[idx:], 'b--')
ax_noise.plot(sig2[idx_mine:], 'r--')


ax_corr.set_title('C factor')
ax_corr.plot(lags, corr, 'b')
ax_corr.plot(idx, 1, 'bx', ms=10)
ax_corr.grid('on')
ax_mine.set_title('C factor')

ax_mine.plot(compute_times, np.array(CS/max(CS)), 'r')
ax_mine.plot(compute_times[idx_mine], 1, 'rx', ms=10)
ax_mine.plot(idx, 1, 'ro', ms=10)

ax_mine.grid('on')
fig.tight_layout()
plt.show()

