"""
Program: figure.py
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def rc_params():
    """Set rcParams for this plot"""
    params = {
        'axes.linewidth':2.0,
        'xtick.major.size':6.5,
        'xtick.major.width':1.5,
        'xtick.minor.size':3.5,
        'xtick.minor.width':1.5,
        'ytick.major.size':6.5,
        'ytick.major.width':1.5,
        'ytick.minor.size':3.5,
        'ytick.minor.width':1.5,
        'xtick.labelsize':22,
        'ytick.labelsize':22,
        'xtick.direction':'in',
        'ytick.direction':'in',
        'xtick.top':True,
        'ytick.right':True
        }
    mpl.rcParams.update(params)

# Set rcParams    
rc_params()

# Data
cwd = os.getcwd()
data_path = os.path.join(cwd, 'results.dat')

# Figure
fig_path  = os.path.join(cwd, 'figure.png')

# Load data
darr = np.loadtxt(data_path, skiprows=2)
t = darr[:,0]
y = darr[:,1]
y_ex = darr[:,2]

# Plot results
fig, ax = plt.subplots(figsize=(8,6))

p1, = ax.plot(t, y_ex, linewidth = 3.0, color="red", alpha=0.5,
              linestyle='--', label='Exact Solution')

p2, = ax.plot(t, y_ex, linewidth = 3.0, color="blue", alpha=0.5,
              marker='o', markersize=7, linestyle='', label='Numeric Solution')


plt.xlim([0.0, 5.0])
plt.ylim([0.0, 4.3])
plt.xlabel('Time', fontsize=22)
plt.ylabel('u(t)', fontsize=22)
plt.legend(fontsize=15, loc="upper left", shadow=True, fancybox=True)

plt.savefig(fig_path, format='png', dpi=100, bbox_inches='tight')

