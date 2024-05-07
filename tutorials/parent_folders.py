import sys
sys.path.append('../')


import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams['grid.linewidth'] = .1

plt.rc('text', usetex=True)
plt.rc('font', family='times', size=14, serif='Times New Roman')
plt.rc('mathtext', rm='times', bf='times:bold')
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')