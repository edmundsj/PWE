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
from core.matrices import *
from netlist.netlist_parser import *
import matplotlib.pyplot as plt

# 1. The class NetlistParser parses a netlist and turns everything into a "Mask" or "Field" object.
# The masks and field are returned so that they are sorted in ascending order with
# respect to their coordinate on the optical axis.
arguments = len(sys.argv) - 1; # The number of arguments
netlist_location = './netlist/sample_netlist.txt';
print(arguments);
print(sys.argv);
if(arguments > 0):
    print(f"Using user defined netlist {sys.argv[1]}")
    netlist_location = sys.argv[1];
print("Parsing netlist... ");
parser = NetlistParser(netlist_location);

# Up until this point, everything is totally general, but now we have to decide
# what it is our parser returns. I will try to have a consistent netlist format across
# all my codes to the degree that is possible.
[er, ur, t, sources] = parser.parseNetlist();
print(f"Done. Found:\nPermittivities:{er}\nPermeabilities:{ur}\nInternal Layers: {t}")

# First, figure out how many layers we have
num_internal_layers = len(t);
if(len(er) != num_internal_layers + 2):
    raise Exception(f"Error: The number of layers is not equal to the number of permittivities. Number of permittivities is {len(er)} and number of layers is {num_layers+2}");

if(len(ur) != num_internal_layers + 2):
    raise Exception(f"Error: The number of layers is not equal to the number of permeabilities. Number of permeabilities is {len(ur)} and number of layers is {num_layers+2}");


print("Initializing Simulation... Setting up materials, polarization, incident wave...")
# Setup material parameters used in the simulation
