from StateMachine.StateMachine import State
from StateMachine.Context import contextDoE
from Common import Common
from Common import Statistics
from Common import CombinationFactory
from Common import LinearRegression as LR

import numpy as np

import logging

context = None


class InitDoE(State):
    def __init__(self):
        super().__init__("Initialize DoE")
        
    def onCall(self):

        global context

        context = contextDoE()
        logging.info(str(context.factorSet))

        return FindNewExperiments()


class FindNewExperiments(State):
    def __init__(self):
        super().__init__("Find new experiments")

    def onCall(self):

        experiments = context.experimentFactory.getNewExperimentSuggestion()
        context.newExperimentValues = context.factorSet.realizeExperiments(experiments, sortColumn=0)

        return ExecuteExperiments()

class ExecuteExperiments(State):
    def __init__(self):
        super().__init__("Execute experiments")

    def onCall(self):

        Y = np.array([x.getValueArray() for x in context.xamControl.workOffExperiments(context.newExperimentValues)])
        context.addNewExperiment(context.newExperimentValues, Y)

        return EvaluateExperiments()


class EvaluateExperiments(State):
    def __init__(self):
        super().__init__("Evaluate experiments")

    def getInitCombinations(self):
        return CombinationFactory.allLinearCombinations() # CombinationFactory.allCombinations() # 

    def createModels(self, combinations):

        context.scaledModel = Common.getModel(context.experimentValues, combinations, context.Y[:, 0], Statistics.orthogonalScaling)
        context.model = Common.getModel(context.experimentValues, combinations, context.Y[:, 0])

        return context.scaledModel, context.model

    def removeLeastSignificantCombination(self, combinations, conf_int):
        _, significanceInterval = Statistics.getModelTermSignificance(conf_int)
        significanceInterval[0:5] = 100
        return Common.removeCombinations(combinations, lambda index, k, v: index == np.argmin(significanceInterval)) 

    def getCombinationsForBinPattern(self, combinationSet, number):
        removeList = list(range(5))
        removeList.extend([int(d) for d in str(bin(number))[2:]])
        removeList.extend(np.zeros(len(combinationSet)))

        return Common.removeCombinations(combinationSet, lambda index, k, v: removeList[index] > 0) 

    def onCall(self):

        combinations = self.getInitCombinations()

        iterationIndex, iterationHistory = 0, {}
        while len(combinations) > 0:
            scaledModel, _ = self.createModels(combinations)

            X = Common.getXWithCombinations(context.experimentValues, combinations, Statistics.orthogonalScaling)
            trainingY, predictionY = context.Y[:, 0], LR.predict(scaledModel, X)

            r2Score = Statistics.R2(trainingY, predictionY)
            q2Score = Statistics.Q2(X, trainingY, predictionY)
            
            iterationHistory[iterationIndex] = (combinations, r2Score, q2Score)
            iterationIndex+=1

            combinations = self.removeLeastSignificantCombination(combinations, scaledModel.conf_int())

        

        scoreIndex = 1
        getScoreOfSet = lambda setItem: setItem[1][scoreIndex]
        maxR2Score =  getScoreOfSet(max(iterationHistory.items(), key=getScoreOfSet))
        filteredScoreHistory = dict(filter(lambda e: getScoreOfSet(e) > .95*maxR2Score, iterationHistory.items()))
        filtered = min(filteredScoreHistory.items(), key=lambda a: len(a[1][0]))

        selctedIndex, (combinations, r2Score, q2Score) = filtered
        scaledModel, _ = self.createModels(combinations)
        context.factorSet.setExperimentValueCombinations(combinations)
        
        Statistics.plotR2ScoreHistory([a[scoreIndex] for a in iterationHistory.values()],selctedIndex)
        Statistics.plotCoefficients(scaledModel.params, context.factorSet, scaledModel.conf_int())

        Statistics.plotObservedVsPredicted(LR.predict(scaledModel, Common.getXWithCombinations(context.experimentValues, combinations, Statistics.orthogonalScaling)), context.Y[:, 0])
        Statistics.plotResiduals(Statistics.residualsDeletedStudentized(scaledModel))

        return HandleOutliers()


class StopDoE(State):
    def __init__(self):
        super().__init__("Stop DoE")

    def onCall(self):
        return None


class HandleOutliers(State):
    def __init__(self):
        super().__init__("Handle outliers")

    def detectOutliers(self):
        outlierIdx = context.scaledModel.outlier_test()[:, 0] > 4
        return outlierIdx

    def countExperimentsInRange(self, outlierIdx, range = .05):
        X = Statistics.orthogonalScaling(context.experimentValues)
        diff = np.sqrt(((X - X[outlierIdx,:])**2).sum(axis=1))
        return (diff < range).sum()

    def forEachOutlier(self):
        for idx, isOultier in enumerate(self.detectOutliers()):
            if isOultier: yield (idx, context.experimentValues[idx, :])

    def executeNewExperimentsAroundOutlier(self, outlier):
        newExperimentValues = np.array([outlier]) #np.array([outlier, outlier])
        return np.array([x.getValueArray() for x in context.xamControl.workOffExperiments(newExperimentValues)]), newExperimentValues
  

    def onCall(self):

        if not any(self.detectOutliers()): 
            print("No outliers detected")
            return StopDoE()

        for (idx, outlier) in self.forEachOutlier():
                
            similarExperimentCount = self.countExperimentsInRange(idx)
            print("For Outlier #{} ({}) there are {} similar experiment(s)".format(idx, str(outlier), similarExperimentCount-1))

            if similarExperimentCount <= 1:
                repeatedY, newExperimentValues = self.executeNewExperimentsAroundOutlier(outlier)
                repeatLimit = .1

                # TODO - remove mock
                repeatedY = np.array([.1, repeatedY[0, 1]])

                if all((abs(context.Y[idx, :] - repeatedY) < repeatLimit).reshape(-1, 1)):
                    # All the same again:
                    print("For Outlier #{} ({}) one more measurements resulted in the same outcome -> Remove both".format(idx, str(outlier)))
                    context.deleteExperiment(idx)

                else:
                    # new result:
                    print("For Outlier #{} ({}) one more measurements resulted in a different outcome -> Replace them".format(idx, str(outlier)))
                    context.deleteExperiment(idx)
                    context.addNewExperiment(newExperimentValues, np.array([repeatedY]))

                
                return EvaluateExperiments()


