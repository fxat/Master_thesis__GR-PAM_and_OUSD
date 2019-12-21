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

from simPAMsig import photoAcousticSignal

#####[ SIMULATION ]############################################################

def main():
    sim = photoAcousticSignal()
    
    """
    Create a N-shaped preasure function that is excited 
    by an ideal laser pulse
    """
    sim.deltaExciteSphere()

    """
    The convolution of the ideal N-shaped signal with the gaussian 
    shaped laser pulse gives a realistic source signal
    """
    sim.gaussianProfileLaserPulse()

    """ 
    The transfer function of a typical ultrasonic detector is 
    gaussian shaped. Therefore a gaussian shaped function in the 
    frequency domain is constructed
    """
    sim.sphereExciteLaser()

    """ 
    By multiplying the fourier transformed signal of the Sphere 
    with the complex transfer funtion of the sensor, the result is the 
    spectrum of the measured signal.
    """  
    sim.spectrumSphereSig() 

    """
    By performing the inverse fourier transformation, we get the 
    temporal measurement signal of the sensor 
    """    
    sim.sensorTransferFunction()

    """ 
    Resulting signal spectrum that can be measured at the transducer 
    """
    sim.resTransducerSig()

    """ 
    Resulting temporal signals for a point like detector and a spherical
    detector
    """
    sim.resTempSigPointSphericalSource()
    
    sim.plotSim()

if __name__ == "__main__":
    main()
