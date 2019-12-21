#!/usr/bin/env python

__author__ = "Franz Taffner"
__copyright__ = "Copyright 2019"
__license__ = "MIT"
__version__ = "0.1"
__email__ = "franz.taffner@yahoo.de"

"""
This script simulatates a excitation of a acoustical wave by a laser. 
Furthermore the detection with a pointlike detector and a bandwidth limited
sensor is examined.
"""

#####[ IMPORTS ]###############################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sps

import constants as C
from plots.simPlots import pamPlots

#####[ CLASS ]#################################################################

class photoAcousticSignal:

    def __init__(self):
        self.p0 = C.F * C.mu_a * C.gamma   # Initial preasure rise

        self.t = np.linspace(0, C.t_max, C.N)    # Time intervall
        self.freq = np.linspace(0, (C.N)/np.max(self.t), len(self.t)) # Fequency intervall
        # Create frequency interval for real-valued signals
        self.freq = self.freq - np.max(self.freq)/2   

        self.dist = C.cs * self.t # Convert time domain into a spatial domain

        self.pamSimPlots = pamPlots() 

    def deltaExciteSphere(self):
        """ Delta pulse excitation of a shere """

        data = {'dist': C.cs*self.t, 'preasure': np.zeros(C.N)}
        self.pP = pd.DataFrame(data, dtype=float)

        """
        Create a N-shaped preasure function that is excited 
        by an ideal laser pulse
        """
        self.pP['preasure']= np.piecewise(self.t, 
                        [np.logical_and(C.cs*self.t>(C.z-C.a), C.cs*self.t<(C.z+C.a))], 
                        [lambda tt: 0.5*(C.z-C.cs*tt)/C.z])

        ax_pP = self.pP.plot(x='dist', y='preasure')
        self.pamSimPlots.plot_pP(ax_pP)

    def gaussianProfileLaserPulse(self):
        """ Gaussian temporal profil of a excitation laser pulse """

        # Shape a gaussian function in the temporal domain ~ laster pulse
        sigma = C.tp/(2*np.sqrt(2*np.log(2)))
        LP = np.exp(-(self.t-C.t0)**2/(2*sigma**2))

        t_shift = self.t-C.t_max/2
        laserPulseData = {'lPtime': t_shift, 'lPpowerNormed': LP}

        self.laserPulse = pd.DataFrame(laserPulseData, dtype=float)

        ax_laserPulse = self.laserPulse.plot(x='lPtime', y='lPpowerNormed')
        self.pamSimPlots.plot_laserPulse(ax_laserPulse)

    def sphereExciteLaser(self):
        """ 
        Signal of a sphere with finite excitation pulse duration.
        """

        sigSpherePulse = np.convolve(self.pP['preasure'], 
                                    self.laserPulse['lPpowerNormed'], 
                                    mode='same')
        sigSphereData = {'dist': self.dist, 'sigSphere': sigSpherePulse}
        self.sigSphere = pd.DataFrame(sigSphereData, dtype=float)

        #ax_sigSphere = self.sigSphere.plot(x='dist', y='sigSphere')
        #self.pP.plot(x='dist', y='preasure', ax=ax_sigSphere)
        #self.pamSimPlots.plot_sigSphere(ax_sigSphere)

    def spectrumSphereSig(self):
        """ Spectrum of the spherical signal """

        fourierT_sigSphere = np.fft.fftshift(np.fft.fft(self.sigSphere['sigSphere']))
        absFFT_sigSphere = np.abs(fourierT_sigSphere)
        absFFT_sigSphere = absFFT_sigSphere/np.max(absFFT_sigSphere)    # Normalize sprectra

        fftSigSphereNorm = absFFT_sigSphere/np.max(absFFT_sigSphere)

        fft_sigSphereData = {'freq': self.freq, 
                            'fftSigSphere': fourierT_sigSphere, 
                            'absfftSigSphere': absFFT_sigSphere, 
                            'fftSigSphereNorm': fftSigSphereNorm}
        self.fft_sigSphere = pd.DataFrame(fft_sigSphereData)

        #ax_fftSigSphere = self.fft_sigSphere.plot(x='freq', y='absfftSigSphere')
        #self.pamSimPlots.plot_fftSigSphere(ax_fftSigSphere)

    def sensorTransferFunction(self):
        """ Construct sensor transfer function """

        sensor_sigma = C.sensor_bw/(2*np.sqrt(2*np.log(2)))
        # Sensor amplitude spectrum
        sensor_as = np.exp(-(np.abs(self.freq)-C.sensor_fc)**2/(2*sensor_sigma**2)) 

        # To avoid the log(0) -> all 0 are replaced by a min value 
        fract = 100
        sensor_as[np.where(sensor_as < np.max(sensor_as)/fract)] = np.max(sensor_as)/fract

        # Sensor response functions phase
        sensor_phi = np.imag(sps.hilbert(np.log(sensor_as)))
        # Complex transfer function of the sensor
        sensor_cf = sensor_as*(np.cos(sensor_phi) - 1j*np.sin(sensor_phi))

        sensorData = {'freq': self.freq, 'cf': sensor_cf, 'as': sensor_as}
        self.sensor = pd.DataFrame(sensorData)

    def resTransducerSig(self):
        """ Resulting signal spectrum that can be measured at the transducer """
        
        specResultTransducer = self.fft_sigSphere['fftSigSphere'] * self.sensor['cf']
        specResultNorm = np.abs(specResultTransducer)/np.max(np.abs(specResultTransducer))

        specResData = {'freq': self.freq, 
                        'specTransMeas': specResultTransducer, 
                        'specTransNorm': specResultNorm}
        self.transducerSig = pd.DataFrame(specResData)

        ax_sensor = self.sensor.plot(x='freq', y='as')
        self.transducerSig.plot(x='freq', y='specTransNorm', ax=ax_sensor)
        self.fft_sigSphere.plot(x='freq', y='fftSigSphereNorm', ax=ax_sensor) 
        self.pamSimPlots.plot_sensor(ax_sensor)

    def resTempSigPointSphericalSource(self):
        """ 
        Resulting temporal signals for a point like detector and a spherical
        detector
        """

        specResultTransducer_temp = np.fft.ifft(np.fft.ifftshift(self.transducerSig['specTransMeas'])).real
        
        specResultTransducerData_temp = {'dist': self.dist, 
                                        'tempSig': self.p0*specResultTransducer_temp,
                                        'sigSphere': self.p0*self.sigSphere['sigSphere']}
        self.transducerSig_temp = pd.DataFrame(specResultTransducerData_temp)

        ax_transSigTemp = self.transducerSig_temp.plot(x='dist', y='tempSig')
        self.transducerSig_temp.plot(x='dist', y='sigSphere', ax=ax_transSigTemp)
        self.pamSimPlots.plot_transSigTemp(ax_transSigTemp)

    def plotSim(self):
        plt.show()