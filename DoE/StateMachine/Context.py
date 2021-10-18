from XamControl import XamControl
from Common import ExperimentFactory
from Common import Factor

import numpy as np

class contextDoE():

    def __init__(self):

        self.xamControl = XamControl.XamControlModdeYMock()
        self.experimentFactory = ExperimentFactory.ExperimentFactory()
        self.factorSet = Factor.getDefaultFactorSet()
        
        self.newExperimentValues = np.array([])
        self.experimentValues = np.array([])

        self.Y = np.array([])

        self.model = None
        self.scaledModel = None


    def addNewExperiment(self, newExperimentValues, Y):
        gAppend = lambda x_, n_: n_ if len(x_) <= 0 else np.append(x_, n_, axis=0)

        self.experimentValues = gAppend(self.experimentValues, newExperimentValues)
        self.Y = gAppend(self.Y, Y)

    def deleteExperiment(self, idx):
        self.Y = np.delete(self.Y, idx, axis=0)
        self.experimentValues = np.delete(self.experimentValues, idx, axis=0)