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
    sim.deltaExciteSphere()
    sim.gaussianProfileLaserPulse()
    sim.sphereExciteLaser()  
    sim.spectrumSphereSig()
      
    sim.sensorTransferFunction()
    sim.plotSim()
    #sim.resTransducerSig()
    #sim.plotSim()
    #sim.resTempSigPointSphericalSource()
    #sim.plotSim()

if __name__ == "__main__":
    main()
