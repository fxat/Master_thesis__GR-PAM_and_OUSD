#!/usr/bin/env python

__author__ = "Franz Taffner"
__copyright__ = "Copyright 2019"
__license__ = "MIT"
__version__ = "0.1"
__email__ = "franz.taffner@yahoo.de"

"""

"""

#####[ IMPORTS ]###############################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sps

from .. import constants as C

class pamPlots:

    def plot_pP(self, ax):
        ax.set_title('Pressure signal: $\delta$-peak excitation')
        ax.set_xlabel('Distance in m')
        ax.set_ylabel('Preassure in Pa')
        ax.set_xlim((C.z-5*C.a), (C.z+5*C.a))
        ax.grid()

    def plot_laserPulse(self, ax):
        ax.set_title('Temporal profile of a laser pulse')
        ax.set_xlabel('Time in s')
        ax.set_ylabel('Laser power in N.A.')
        ax.set_xlim(-2*tp, 2*tp)
        ax.grid()

    def plot_sigSphere(self, ax):
        ax.set_title('Ideal measured signal')
        ax.set_xlabel('Distance in m')
        ax.set_ylabel('N.A.')
        ax.set_xlim(C.z-3*C.a, C.z+3*C.a)
        ax.grid()

    def plot_fftSigSphere(self, ax):
        ax.set_title('Frequency spectrum of an ideal Signal')
        ax.set_xlabel('f in Hz')
        ax.set_ylabel('N.A')
        ax.set_xlim(-200e6, 200e6)
        ax.grid()

    def plot_sensor(self, ax):
        ax.set_title('Frequency domain representation')
        ax.legend(['Pressure wave spectrum', 'Resulting signal spectrum', 'Transducer transferfunction'])
        ax.set_xlabel('f in Hz')
        ax.set_ylabel('N.A.')
        ax.set_xlim(-200e6, 200e6)
        ax.grid()

    def plot_transSigTemp(self, ax):
        ax.set_title('Comparison of point detector temporal signal with \n spherical detector signal')
        ax.legend(['Bandwidth limited detector', 'Pointlike detector'])
        ax.set_xlabel('$c_st$ in m')
        ax.set_ylabel('p in Pa')
        ax.set_xlim(C.z-10*C.a, C.z+50*C.a)
        ax.grid()
    