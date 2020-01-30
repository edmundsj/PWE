# Author: Jordan Edmunds, Ph.D. Student, UC Berkeley
# Contact: jordan.e@berkeley.edu
# Creation Date: 11/01/2019
#
# TODO:
# Fix netlist parser so it can handle zero layers and assume the input and output layers are free space
# (Eventually, maybe never):
# - relax the constraint that the transmitting and incident medium must be LHI materials.

import numpy as np
import scipy as sp
import scipy.linalg
import sys

sys.path.append('core')

from matrices import *
from netlist.netlist_parser import *
import matplotlib.pyplot as plt


# For this program, I'm not certain how to easily generate a netlist. Probably just store permittivity and
# permeability tensors in a 2D array, or as a function of position. It's kind awkward though. Might need to 
# use some type of structural information.

a = 0.41;
ax = a;
ay = a;
r = 0.25 * a;
er = 2.2*2.2;

t1 = np.array([ax, 0]);
t2 = np.array([0, ay]);

Nx = 512;
Ny = 512;
dx = ax / Nx;
dy = ay / Ny;

xcoors = np.linspace(-ax/2 + dx/2, ax/2 - dx/2, Nx);
ycoors = np.linspace(-ay/2 + dy/2, ay/2 - dy/2, Ny);
(X, Y) = np.meshgrid(xcoors, ycoors);
UR = complexOnes((Nx, Ny));
ER = (er-1) * np.heaviside(sq(X) + sq(Y) - sq(r),1)
ER = ER + 1;

pointsPerWalk = 10;
crystal = Crystal(ER, UR, t1, t2)
band = BandStructure(crystal)
numberInternalPoints = 0;
numberHarmonics = (21, 21,1)
band.Solve(numberHarmonics, numberInternalPoints)
band.Plot(0.9)
