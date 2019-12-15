#!/usr/bin/env python

__author__ = "Franz Taffner"
__copyright__ = "Copyright 2019"
__license__ = "MIT"
__version__ = "0.1"
__email__ = "franz.taffner@yahoo.de"

"""

"""

#%%
#####[ IMPORTS ]###############################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sps

#%%
#####[ DEFINES ]###############################################################

N = int(1e5) # Number of datapoints

cs = 1500       # Speed of sound in water [m/s]
a = 10e-6       # Radius of a spherical source [m]
z = 10e-3       # Distance source <-> detector [m]

F = 200         # Fluence [J/m]
mu_a = 20000    # Absorption coefficient [1/m]
gamma = 0.11    # Grueneisen Parameter

#####[ PRECONDITIONS ]#########################################################

p0 = F * mu_a * gamma   # Initial preasure rise

t_max = 2 * z / cs
t = np.linspace(0, t_max, N)    # Time intervall
dt = t[1] - t[0]
freq = np.linspace(0, (N-1)/np.max(t), len(t))
freq = freq - np.max(freq)/2

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

tp = 10e-9      # Temporal laser pulse width [ns] 
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

fourierT_sigSphere = np.fft.fftshift(np.fft.fft(sigSphere['sigSphere']))
absFFT_sigSphere = np.abs(fourierT_sigSphere)
absFFT_sigSphere = absFFT_sigSphere/np.max(absFFT_sigSphere)    # Normalize sprectra

fftSigSphereNorm = absFFT_sigSphere/np.max(absFFT_sigSphere)

fft_sigSphereData = {'freq': freq, 'fftSigSphere': fourierT_sigSphere, 'absfftSigSphere': absFFT_sigSphere, 'fftSigSphereNorm': fftSigSphereNorm}
fft_sigSphere = pd.DataFrame(fft_sigSphereData, dtype=float)

ax_fftSigSphere = fft_sigSphere.plot(x='freq', y='absfftSigSphere')
ax_fftSigSphere.set_title('Frequency spectrum of an ideal Signal')
ax_fftSigSphere.set_xlabel('f in Hz')
ax_fftSigSphere.set_ylabel('N.A')
ax_fftSigSphere.set_xlim(-200e6, 200e6)
ax_fftSigSphere.grid()

#%% Construct sensor transfer function

sensor_fc = 50e6    # Sensor center frequency [Hz]
sensor_bw = 0.7 * sensor_fc # Sensor bandwidth [Hz]

sensor_sigma = sensor_bw/(2*np.sqrt(2*np.log(2)))
sensor_as = np.exp(-(np.abs(freq)-sensor_fc)**2/(2*sensor_sigma**2))   # Sensor amplitude spectrum

# To avoid the log(0) -> all 0 are replaced by a min value 
fract = 100
sensor_as[np.where(sensor_as < np.max(sensor_as)/fract)] = np.max(sensor_as)/fract

# Sensor response functions phase
sensor_phi = np.imag(sps.hilbert(np.log(sensor_as)))  

sensor_cf = sensor_as*(np.cos(sensor_phi) - 1j*np.sin(sensor_phi))

sensorData = {'freq': freq, 'cf': sensor_cf, 'as': sensor_as}
sensor = pd.DataFrame(sensorData, dtype=float)

# Resulting signal spectrum that can be measured at the transducer
specResultTransducer = fourierT_sigSphere * sensor['cf'] 
specResultNorm = np.abs(specResultTransducer)/np.max(np.abs(specResultTransducer))

specResData = {'freq': freq, 'specTransMeas': specResultTransducer, 'specTransNorm': specResultNorm}
transducerSig = pd.DataFrame(specResData, dtype=float)

# TODO: fix specTransNorm

ax_sensor = sensor.plot(x='freq', y='as')
transducerSig.plot(x='freq', y='specTransNorm', ax=ax_sensor)
fft_sigSphere.plot(x='freq', y='fftSigSphereNorm', ax=ax_sensor) 
ax_sensor.set_xlim(-200e6, 200e6)





print('x')

#%%

plt.show()


