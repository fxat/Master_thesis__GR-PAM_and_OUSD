#!/usr/bin/env python

__author__ = "Franz Taffner"
__copyright__ = "Copyright 2019"
__license__ = "MIT"
__version__ = "0.1"
__email__ = "franz.taffner@yahoo.de"

"""

"""

#%%
#####[ IMPORTS ]#########################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#####[ DEFINES ]#########################################################

N = int(1e5) # Number of datapoints

cs = 1500       # Speed of sound in water [m/s]
a = 10e-6       # Radius of a spherical source [m]
z = 10e-3       # Distance source <-> detector [m]

F = 200         # Fluence [J/m]
mu_a = 20000    # Absorption coefficient [1/m]
gamma = 0.11    # Grueneisen Parameter

#####[ PRECONDITIONS ]####################################################

p0 = F * mu_a * gamma   # Initial preasure rise

t_max = 2 * z / cs
t = np.linspace(0, t_max, N)    # Time intervall
dt = t[1] - t[0]
#freq = np.linspace(0, 1/np.max(t), (N-1)/np.max(t))
#freq = freq - np.max(freq)/2

#%% Delta pulse excitation of a shere

data = {'dist': cs*t, 'preasure': np.zeros(N)}
pP = pd.DataFrame(data, dtype=float)

pP['preasure']= np.piecewise(t, [np.logical_and(cs*t > (z-a), cs*t <(z+a))], [lambda t: 0.5*(z-cs*t)/z])

ax_pP = pP.plot(x='dist', y='preasure')
ax_pP.set_xlim((z-5*a), (z+5*a))
ax_pP.set_title('Pressure signal: $\delta$-peak excitation')
ax_pP.set_xlabel('Distance in m')
ax_pP.set_ylabel('Preassure in Pa')
ax_pP.grid()

#%% Gaussian temporal profil of a excitation laser pulse

tp = 10e-9
t0 = t_max/2

sigma = tp/(2*np.sqrt(2*np.log(2)))
LP = np.exp(-(t-t0)**2/(2*sigma**2))

t_shift = t-t_max/2

laserPulseData = {'lPtime': t_shift, 'lPpowerNormed': LP}

laserPulse = pd.DataFrame(laserPulseData, dtype=float)
ax_laserPulse = laserPulse.plot(x='lPtime', y='lPpowerNormed')
ax_laserPulse.set_title('Temporal profile of a laser pulse')
ax_laserPulse.set_xlabel('Time in s')
ax_laserPulse.set_ylabel('Laser power in N.A.')
ax_laserPulse.set_xlim(-2*tp, 2*tp)
ax_laserPulse.grid()

#%% Signal of a sphere with finite excitation pulse duration

sigSpherePulse = np.convolve(pP['preasure'], laserPulse['lPpowerNormed'], mode='same')
sigSphereData = {'dist': cs*t, 'sigSphere': sigSpherePulse}
sigSphere = pd.DataFrame(sigSphereData, dtype=float)

ax_sigSphere = sigSphere.plot(x='dist', y='sigSphere')
pP.plot(x='dist', y='preasure', ax=ax_sigSphere)
ax_sigSphere.set_title('Ideal measured signal')
ax_sigSphere.set_xlabel('Distance in m')
ax_sigSphere.set_ylabel('N.A.')
ax_sigSphere.set_xlim(z-3*a, z+3*a)
ax_sigSphere.grid()

#%% Spectrum of the spherical signal



print('x')

#%%

plt.show()


def main():
    pass

if __name__ == '__main__':   
    main()

# %%
