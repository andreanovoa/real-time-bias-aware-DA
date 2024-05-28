
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

x = np.arange(1500) / 100

max_len = 500
sig = 2*np.sin(np.pi * x)
sig2 = np.sin(2*(np.pi * x))

max_len = 500
corr = signal.correlate(sig2[:max_len], sig[:max_len])
lags = signal.correlation_lags(max_len, max_len)
zero_lag = np.argmin(abs(lags))
lags = lags[zero_lag:]
corr = corr[zero_lag:]
idx = lags[np.argmax(corr)]


def C(y1, y2):
    return np.sum((y1 * y2)) / np.sqrt(np.sum(y1 ** 2) * np.sum(y2 ** 2))


compute_times = np.linspace(0, len(sig)//2,  len(sig)//2, dtype=int)
CS = [C(sig[:max_len], sig2[:max_len])]
for ii in compute_times[1:]:
    sig2_corr = sig2[ii:max_len+ii]
    CS.append(C(sig[:max_len], sig2_corr))

idx_mine = np.argmax(CS)
print(min(CS), max(CS))

fig, (ax_orig, ax_noise, ax_corr, ax_mine) = plt.subplots(4, 1, sharex=False)
ax_orig.set_title('OG')
ax_orig.plot(sig, 'k', lw=2)
ax_orig.plot(sig2, '--')
ax_noise.set(title='Correlated', xlim=[0, max_len])
ax_noise.plot(sig,  'k', lw=2)

ax_noise.plot(sig2[idx:], 'b--')
ax_noise.plot(sig2[idx_mine:], 'r--')

ax_corr.set_title('C factor')
ax_corr.plot(lags, corr, 'b')
ax_corr.plot(idx, 1, 'bx', ms=10)
ax_corr.grid('on')
ax_mine.set_title('C factor')

ax_mine.plot(compute_times, np.array(CS), 'r')
ax_mine.plot(compute_times[idx_mine], CS[idx_mine], 'rx', ms=10)
ax_mine.plot(idx, 1, 'ro', ms=10)

ax_mine.grid('on')
fig.tight_layout()
plt.show()

