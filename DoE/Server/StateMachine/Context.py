from XamControl import XamControl
from Common import ExperimentFactory
from Common import Transform
from Common import History
from Common import Factor

from Common import LinearRegression as LR

import numpy as np

class ContextDoE():

    def __init__(self, setFactorSet=None, optimum=None, optimumRange=10, returnAllExperimentsAtOnce=False, setXAMControl=None, previousResult=None):

        self.xamControl = XamControl.XamControl() #
        if setXAMControl is not None: self.xamControl = setXAMControl
        
        self.experimentFactory = ExperimentFactory.ExperimentFactory()

        self.factorSet = Factor.getDefaultFactorSet()
        if setFactorSet is not None:
            self.factorSet = setFactorSet
            
        if optimum is not None:
            self.factorSet = Factor.getFactorSetAroundOptimum(self.factorSet, optimum, optimumRange)
        
        self.newExperimentValues = np.array([])
        self._experimentValues = np.array([])

        self.Y = np.array([])

        self.model = None
        self.scaledModel = None

        #self.history = History.History()

        self.excludedFactors = []
        self.deletedExperiments = []
        self.predictedResponses = []

        self.returnAllExperimentsAtOnce = returnAllExperimentsAtOnce
        self.previousResult = previousResult

        self.transformer = None #Transform.BoxCoxOffsetTransformer()

    def getResponse(self, responseIdx=0):
        Y = self.Y[:, responseIdx]

        if self.transformer is None: return Y
        return self.transformer.transform(Y)


    def addNewExperiments(self, newExperimentValues, Y):
        gAppend = lambda x_, n_: n_ if len(x_) <= 0 else np.append(x_, n_, axis=0)

        self._experimentValues = gAppend(self._experimentValues, newExperimentValues)
        self.Y = gAppend(self.Y, Y)

    def addNewPredictedResponses(self, newPredictedResponses):
        gAppend = lambda x_, n_: n_ if len(x_) <= 0 else np.append(x_, n_, axis=0)
        self.predictedResponses = gAppend(self.predictedResponses, newPredictedResponses)

    def deleteExperiment(self, idx):
        self.deletedExperiments.append((self._experimentValues[idx, :], self.Y[idx, :]))

        self.Y = np.delete(self.Y, idx, axis=0)
        self._experimentValues = np.delete(self._experimentValues, idx, axis=0)

    def canPredict(self):
        return self.previousResult is not None

    def predictResponse(self, newExperimentValues):
        if not self.canPredict(): return None

        newExperimentValues = np.delete(newExperimentValues,  self.previousResult.context.excludedFactors, axis=1) 

        X = None
        for exp in newExperimentValues:
            if X is None:
                X = np.append(exp, [func(exp) for func in self.previousResult.combinations.values()])
            else:
                X = np.vstack((X, np.append(exp, [func(exp) for func in self.previousResult.combinations.values()])))

        return LR.predict(self.previousResult.model, X)

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


