from XamControl import XamControl
from Common import ExperimentFactory
from Common import History
from Common import Factor

import numpy as np

class contextDoE():

    def __init__(self):

        self.xamControl = XamControl.XamControlTestRun1Mock()
        self.experimentFactory = ExperimentFactory.ExperimentFactory()
        self.factorSet = Factor.getDefaultFactorSet()
        
        self.newExperimentValues = np.array([])
        self._experimentValues = np.array([])

        self.Y = np.array([])

        self.model = None
        self.scaledModel = None

        self.history = History.History()

        self.excludedFactors = []

    def getResponse(self, responseIdx=1):
        return self.Y[:, responseIdx]

    def addNewExperiments(self, newExperimentValues, Y):
        gAppend = lambda x_, n_: n_ if len(x_) <= 0 else np.append(x_, n_, axis=0)

        self._experimentValues = gAppend(self._experimentValues, newExperimentValues)
        self.Y = gAppend(self.Y, Y)

    def deleteExperiment(self, idx):
        self.Y = np.delete(self.Y, idx, axis=0)
        self._experimentValues = np.delete(self._experimentValues, idx, axis=0)

    def getExperimentValues(self):
        return np.delete(self._experimentValues, self.excludedFactors, axis=1)

    def excludeFactor(self, factorIndex):
        if isinstance(factorIndex, list):
            self.excludedFactors.extend(factorIndex)
        else:
            self.excludedFactors.append(factorIndex)

    def resetFactorExlusion(self):
        self.excludedFactors = []

    def activeFactorCount(self):
        return len(self.factorSet) - len(self.excludedFactors)

    def isFactorExcluded(self, factorIndex):
        return factorIndex in self.excludedFactors
