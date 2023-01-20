#!/usr/bin/env python
"""
Program: tex_demo.py
         Text rendering With LaTeX
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# set to use latex fonts
plt.rcParams['text.usetex'] = True

# choose latex font
plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots()

x = np.linspace(-2.0, 2.0, 10000) # The x-values
sigma = np.linspace(0.4, 1.0, 4)  # Some different values of sigma

# Here we evaluate a Gaussians for each sigma
gaussians = [(2*np.pi*s**2)**-0.5 * np.exp(-0.5*x**2/s**2) for s in sigma]
for s,y in zip(sigma, gaussians):
    ax.plot(x, y, lw=1.25, label=r"$\sigma = %3.2f$"%s)

# formula to display
formula = r"$y(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{x^2}{2\sigma^2}}$"
ax.text(0.05, 0.80, formula, transform=ax.transAxes, fontsize=20)

# configure axes labels
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y(x)$", fontsize=18)

# display legend
ax.legend()

# save figure as png
fig.savefig('tex_demo.png')
