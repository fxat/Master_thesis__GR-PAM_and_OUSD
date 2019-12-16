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
import constants as C

from plots.simPlots import pamPlots

#%%

p0 = C.F * C.mu_a * C.gamma   # Initial preasure rise

t = np.linspace(0, C.t_max, C.N)    # Time intervall
freq = np.linspace(0, (C.N)/np.max(t), len(t))
freq = freq - np.max(freq)/2

pamSimPlots = pamPlots()

#%% Delta pulse excitation of a shere

data = {'dist': C.cs*t, 'preasure': np.zeros(C.N)}
pP = pd.DataFrame(data, dtype=float)

pP['preasure']= np.piecewise(t, [np.logical_and(C.cs*t>(C.z-C.a), C.cs*t<(C.z+C.a))], 
                                [lambda t: 0.5*(C.z-C.cs*t)/C.z])

ax_pP = pP.plot(x='dist', y='preasure')
pamSimPlots.plot_pP(ax_pP)

#%% Gaussian temporal profil of a excitation laser pulse


sigma = C.tp/(2*np.sqrt(2*np.log(2)))
LP = np.exp(-(t-C.t0)**2/(2*sigma**2))

t_shift = t-C.t_max/2
laserPulseData = {'lPtime': t_shift, 'lPpowerNormed': LP}

laserPulse = pd.DataFrame(laserPulseData, dtype=float)
ax_laserPulse = laserPulse.plot(x='lPtime', y='lPpowerNormed')
pamSimPlots.plot_laserPulse(ax_laserPulse)

#%% Signal of a sphere with finite excitation pulse duration

sigSpherePulse = np.convolve(pP['preasure'], laserPulse['lPpowerNormed'], mode='same')
sigSphereData = {'dist': C.cs*t, 'sigSphere': sigSpherePulse}
sigSphere = pd.DataFrame(sigSphereData, dtype=float)

ax_sigSphere = sigSphere.plot(x='dist', y='sigSphere')
pP.plot(x='dist', y='preasure', ax=ax_sigSphere)
pamSimPlots.plot_sigSphere(ax_sigSphere)

#%% Spectrum of the spherical signal

fourierT_sigSphere = np.fft.fftshift(np.fft.fft(sigSphere['sigSphere']))
absFFT_sigSphere = np.abs(fourierT_sigSphere)
absFFT_sigSphere = absFFT_sigSphere/np.max(absFFT_sigSphere)    # Normalize sprectra

fftSigSphereNorm = absFFT_sigSphere/np.max(absFFT_sigSphere)

fft_sigSphereData = {'freq': freq, 
                    'fftSigSphere': fourierT_sigSphere, 
                    'absfftSigSphere': absFFT_sigSphere, 
                    'fftSigSphereNorm': fftSigSphereNorm}
fft_sigSphere = pd.DataFrame(fft_sigSphereData, dtype=float)

ax_fftSigSphere = fft_sigSphere.plot(x='freq', y='absfftSigSphere')
pamSimPlots.plot_fftSigSphere(ax_fftSigSphere)

#%% Construct sensor transfer function

sensor_fc = 50e6    # Sensor center frequency [Hz]
sensor_bw = 0.7 * sensor_fc # Sensor bandwidth [Hz]

sensor_sigma = sensor_bw/(2*np.sqrt(2*np.log(2)))
# Sensor amplitude spectrum
sensor_as = np.exp(-(np.abs(freq)-sensor_fc)**2/(2*sensor_sigma**2)) 

# To avoid the log(0) -> all 0 are replaced by a min value 
fract = 100
sensor_as[np.where(sensor_as < np.max(sensor_as)/fract)] = np.max(sensor_as)/fract

# Sensor response functions phase
sensor_phi = np.imag(sps.hilbert(np.log(sensor_as)))

sensor_cf = sensor_as*(np.cos(sensor_phi) - 1j*np.sin(sensor_phi))

sensorData = {'freq': freq, 'cf': sensor_cf, 'as': sensor_as}
sensor = pd.DataFrame(sensorData)

# Resulting signal spectrum that can be measured at the transducer
specResultTransducer = fourierT_sigSphere * sensor['cf']
specResultNorm = np.abs(specResultTransducer)/np.max(np.abs(specResultTransducer))

specResData = {'freq': freq, 
                'specTransMeas': specResultTransducer, 
                'specTransNorm': specResultNorm}
transducerSig = pd.DataFrame(specResData, dtype=float)

ax_sensor = sensor.plot(x='freq', y='as')
transducerSig.plot(x='freq', y='specTransNorm', ax=ax_sensor)
fft_sigSphere.plot(x='freq', y='fftSigSphereNorm', ax=ax_sensor) 
pamSimPlots.plot_sensor(ax_sensor)

#%%

specResultTransducer_temp = np.fft.ifft(np.fft.ifftshift(specResultTransducer)).real
specResultTransducerData_temp = {'dist': C.cs*t, 
                                'tempSig': p0*specResultTransducer_temp,
                                'sigSphere': p0*sigSphere['sigSphere']}
transducerSig_temp = pd.DataFrame(specResultTransducerData_temp, dtype=float)

ax_transSigTemp = transducerSig_temp.plot(x='dist', y='tempSig')
transducerSig_temp.plot(x='dist', y='sigSphere', ax=ax_transSigTemp)
pamSimPlots.plot_transSigTemp(ax_transSigTemp)

#%%

plt.show()