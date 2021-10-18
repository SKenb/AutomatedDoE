from XamControl import XamControl
from Common import ExperimentFactory
from Common import Factor

import numpy as np

class contextDoE():

    def __init__(self):

        self.xamControl = XamControl.XamControlModdeYMock()
        self.experimentFactory = ExperimentFactory.ExperimentFactory()
        self.factorSet = Factor.getDefaultFactorSet()
        
        self.X = np.array([])
        self.Y = np.array([])