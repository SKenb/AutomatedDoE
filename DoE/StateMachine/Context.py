from XamControl import XamControl
from Common import ExperimentFactory
from Common import Transform
from Common import History
from Common import Factor

import numpy as np

class ContextDoE():

    def __init__(self, optimum=None, optimumRange=10, returnAllExperimentsAtOnce=False):

        self.xamControl = XamControl.XamControlTestRun2Mock() # XamControl.XamControl() #
        self.experimentFactory = ExperimentFactory.ExperimentFactory()

        self.factorSet = Factor.getDefaultFactorSet()
        if optimum is not None:
            self.factorSet = Factor.getFactorSetAroundOptimum(self.factorSet, optimum, optimumRange)
        
        self.newExperimentValues = np.array([])
        self._experimentValues = np.array([])

        self.Y = np.array([])

        self.model = None
        self.scaledModel = None

        self.history = History.History()

        self.excludedFactors = []
        self.deletedExperiments = []

        self.returnAllExperimentsAtOnce = returnAllExperimentsAtOnce

    def getResponse(self, responseIdx=0, transformFlagOrTransformer = False):
        Y = self.Y[:, responseIdx]

        if transformFlagOrTransformer is None: return Y

        if isinstance(transformFlagOrTransformer, bool):
            if not transformFlagOrTransformer: return Y
            
            transformer = Transform.getSuggestedTransformer(Y)
        elif isinstance(transformFlagOrTransformer, Transform.Transformer):
            transformer = transformFlagOrTransformer
        else:
            raise Exception("transformFlagOrTransformer can not be used as Flag and is no Transformer :0")

        return transformer.transform(Y)


    def addNewExperiments(self, newExperimentValues, Y):
        gAppend = lambda x_, n_: n_ if len(x_) <= 0 else np.append(x_, n_, axis=0)

        self._experimentValues = gAppend(self._experimentValues, newExperimentValues)
        self.Y = gAppend(self.Y, Y)

    def deleteExperiment(self, idx):
        self.deletedExperiments.append((self._experimentValues[idx, :], self.Y[idx, :]))

        self.Y = np.delete(self.Y, idx, axis=0)
        self._experimentValues = np.delete(self._experimentValues, idx, axis=0)

    def restoreDeletedExperiments(self):
        for deletedExperiment in self.deletedExperiments:
            self.addNewExperiments([deletedExperiment[0]], [deletedExperiment[1]])

        self.deletedExperiments = []

    def getExperimentValues(self):
        return np.delete(self._experimentValues, self.excludedFactors, axis=1)

    def excludeFactor(self, factorIndex):
        if isinstance(factorIndex, list):
            assert all(np.array(factorIndex) >= 0) and all(np.array(factorIndex) <= len(self.factorSet)), "Ups - U want to exclude a factor which we don't know :/"
            self.excludedFactors.extend(factorIndex)

        else:
            assert factorIndex >= 0 and factorIndex <= len(self.factorSet), "Ups - U want to exclude a factor which we don't know 0.o"
            self.excludedFactors.append(factorIndex)

    def resetFactorExlusion(self):
        self.excludedFactors = []

    def activeFactorCount(self):
        return len(self.factorSet) - len(self.excludedFactors)

    def isFactorExcluded(self, factorIndex):
        return factorIndex in self.excludedFactors

    def getFactorSetIndexFromCoefIndex(self, coefIndex):
        factorIndices = np.array(range(len(self.factorSet)))
        factorIndices = np.delete(factorIndices, self.excludedFactors)
        return factorIndices[coefIndex]


