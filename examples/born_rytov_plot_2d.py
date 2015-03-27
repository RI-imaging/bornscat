#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Compare Born and Rytov approximation

This script creates a colorfull plot.
"""
from __future__ import division
from __future__ import print_function

from matplotlib import pylab as plt
from matplotlib import cm
import numpy as np
import os
import sys
import time

DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, DIR+"/../")

import bornscat

rfac = 4
# Set measurement parameters
# Compute scattered field from cylinder
radius = 10 # wavelengths
nmed = 1.333
ncyl = 1.343
size = 128*rfac # pixels
res = 2*rfac #23 # px/wavelengths


# create refractive index map for Born
n = nmed * np.ones((size,size))
n0 = 1*n
rad = radius*res
x=np.linspace(-size/2,size/2,size, endpoint=False)
xv = x.reshape(-1,1)
yv = x.reshape(1,-1)
n[np.where((xv**2+yv**2 < rad**2))] = ncyl


# Born
print("Born scattered wave")
born_u0 = bornscat.born_2d(n0, nmed, res)
born_u = bornscat.born_2d(n, nmed, res)
bo = born_u/born_u0

# Rytov
print("Rytov scattered wave")
rytov_u0 = bornscat.rytov_2d(n0, nmed, res)
rytov_u = bornscat.rytov_2d(n, nmed, res)
ro = rytov_u/rytov_u0

bph = np.angle(bo)
bam = np.abs(bo)
rph = np.angle(ro)
ram = np.abs(ro)

pmin = min(bph.min(), rph.min())
pmax = max(bph.max(), rph.max())
amin = min(bam.min(), ram.min())
amax = max(bam.max(), ram.max())


# Plot
fig, axes = plt.subplots(2,2)
axes = axes.flatten()
axes[0].set_title("Born phase")
axes[0].imshow(np.angle(bo), cmap=cm.coolwarm, vmin=pmin, vmax=pmax)
axes[1].set_title("Born amplitude")
axes[1].imshow(np.abs(bo), cmap=cm.gray, vmin=amin, vmax=amax)
axes[2].set_title("Rytov phase")
axes[2].imshow(np.angle(ro), cmap=cm.coolwarm, vmin=pmin, vmax=pmax)
axes[3].set_title("Rytov amplitude")
axes[3].imshow(np.abs(ro), cmap=cm.gray, vmin=amin, vmax=amax)

plt.tight_layout()
plt.savefig(os.path.join(DIR, "born_rytov_plot_2d.png"))
