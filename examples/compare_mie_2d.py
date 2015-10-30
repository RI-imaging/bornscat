#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" TODO

compare scattered field of Rytov with Mie scattering
"""
from __future__ import division
from __future__ import print_function

import numpy as np

import matplotlib
matplotlib.use("wxagg")
from matplotlib import pylab as plt

import os
import sys

import time
import unwrap

DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, DIR+"/../")

try:
    import miefield as mie
except:
    print("Either run 'pip install miefield' or download it manually "+
          "from https://github.com/paulmueller/miefield")
    exit(0)

import bornscat as br

rfac = 4
# Set measurement parameters
# Compute scattered field from cylinder
radius = 5 # wavelengths
nmed = 1.333
ncyl = 1.334
size = 128*rfac # pixels, odd pixels make the day?
res = 2*rfac #23 # px/wavelengths
lambd = 500

# Number of projections
A = 200

# create refractive index map for Born
n = nmed * np.ones((size,size))
n0 = 1*n
rad = radius*res
#x=np.linspace(-size/2,size/2,size, endpoint=False)
x=np.linspace(-size/2,size/2,size, endpoint=False)
xv = x.reshape(-1,1)
yv = x.reshape(1,-1)
n[np.where((xv**2+yv**2 < rad**2))] = ncyl

### 2D plotting born

# Rytov
print("Rytov scattered wave")
rytov_u0 = br.rytov_2d(n0, nmed, res)
rytov_u = br.rytov_2d(n, nmed, res)
ro = rytov_u/rytov_u0

plt.imshow(np.angle(ro))
plt.show()

import IPython
IPython.embed()
