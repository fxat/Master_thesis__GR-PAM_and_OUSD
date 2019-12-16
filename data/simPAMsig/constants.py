#!/usr/bin/env python

__author__ = "Franz Taffner"
__copyright__ = "Copyright 2019"
__license__ = "MIT"
__version__ = "0.1"
__email__ = "franz.taffner@yahoo.de"

"""

"""

N = int(1e5)    # Number of datapoints

cs = 1500       # Speed of sound in water [m/s]
a = 10e-6       # Radius of a spherical source [m]
z = 10e-3       # Distance source <-> detector [m]

F = 200         # Fluence [J/m]
mu_a = 20000    # Absorption coefficient [1/m]
gamma = 0.11    # Grueneisen Parameter

t_max = 2 * z / cs

#%% Laserpulse

tp = 10e-9      # Temporal laser pulse width [ns] 
t0 = t_max/2