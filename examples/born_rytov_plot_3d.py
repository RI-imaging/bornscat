#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Compare Born and Rytov approximation

This script creates a colorfull plot.
"""
from __future__ import division
from __future__ import print_function

from matplotlib import pylab as plt
import numpy as np
import os
import sys

DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, DIR+"/../")

import bornscat
import unwrap

rfac = 1
# Set measurement parameters
# Compute scattered field from cylinder
radius = 3 # wavelengths
nmed = 1.333
nsph = 1.343
size = 64*rfac # pixels
res = 4*rfac #23 # px/wavelengths

fft_method = "pyfftw"

# create refractive index map for Born
n = nmed * np.ones((size,size,size))
n0 = 1*n
rad = radius*res
x=np.linspace(-size/2,size/2,size, endpoint=False)
xv = x.reshape(-1, 1, 1)
yv = x.reshape( 1,-1, 1)
zv = x.reshape( 1, 1,-1)
n[np.where((xv**2+yv**2+zv**2 < rad**2))] = nsph

# Rytov
print("Rytov scattered wave")
rytov_u0 = bornscat.rytov_3d(n0, nmed, res, fft_method=fft_method)
rytov_u = bornscat.rytov_3d(n, nmed, res, fft_method=fft_method)
ro = rytov_u/rytov_u0

rph = np.angle(ro)
rph = unwrap.unwrap(rph)

ram = np.abs(ro)

phakwargs = {"vmin": rph.min(),
             "vmax": rph.max(),
             "cmap": "coolwarm"}

ampkwargs = {"vmin": ram.min(),
             "vmax": ram.max(),
             "cmap": "gray"}

# Plot
fig, axes = plt.subplots(2,3)
axes = axes.transpose().flatten()
axes[0].set_title("Rytov phase z=0")
axes[0].imshow(rph[:,:,size//2], **phakwargs)
axes[1].set_title("Rytov amplitude z=0")
axes[1].imshow(ram[:,:,size//2], **ampkwargs)

axes[2].set_title("Rytov phase y=0")
axes[2].imshow(rph[:,size//2,:], **phakwargs)
axes[3].set_title("Rytov amplitude y=0")
axes[3].imshow(ram[:,size//2,:], **ampkwargs)

axes[4].set_title("Rytov phase x=0")
axes[4].imshow(rph[size//2,:,:], **phakwargs)
axes[5].set_title("Rytov amplitude x=0")
axes[5].imshow(ram[size//2,:,:], **ampkwargs)

plt.tight_layout()
plt.savefig(os.path.join(DIR, "born_rytov_plot_3d.png"))
