from StateMachine.StateMachine import State
from StateMachine.Context import contextDoE
from Common import Common
from Common import Logger
from Common import Statistics
from Common import CombinationFactory
from Common import LinearRegression as LR

import numpy as np

context = None


class InitDoE(State):
    def __init__(self): super().__init__("Initialize DoE")
        
    def onCall(self):

        global context

        context = contextDoE()
        Logger.logStateInfo(str(context.factorSet))

        return FindNewExperiments()


class FindNewExperiments(State):
    def __init__(self): super().__init__("Find new experiments")
    def onCall(self):

        experiments = context.experimentFactory.getNewExperimentSuggestion()
        context.newExperimentValues = context.factorSet.realizeExperiments(experiments, sortColumn=0, sortReverse=len(context.history) % 2)

        return ExecuteExperiments()

class ExecuteExperiments(State):
    def __init__(self): super().__init__("Execute experiments")
    def onCall(self):

        Y = np.array([x.getValueArray() for x in context.xamControl.workOffExperiments(context.newExperimentValues)])
        context.addNewExperiments(context.newExperimentValues, Y)

        return EvaluateExperiments()


class EvaluateExperiments(State):
    def __init__(self): super().__init__("Evaluate experiments")

    def getInitCombinations(self):
        return CombinationFactory.allLinearCombinations() # CombinationFactory.allCombinations() # 

    def createModels(self, combinations, responseIdx = 1):

        context.scaledModel = Common.getModel(context.experimentValues, combinations, context.Y[:, responseIdx], Statistics.orthogonalScaling)
        context.model = Common.getModel(context.experimentValues, combinations, context.Y[:, responseIdx])

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

    def stepwiseRemoveCombinations(self, combinations, responseIdx = 1):
        iterationIndex, iterationHistory = 0, {}

        while len(combinations) > 1:
            scaledModel, _ = self.createModels(combinations)

            X = Common.getXWithCombinations(context.experimentValues, combinations, Statistics.orthogonalScaling)
            trainingY, predictionY = context.Y[:, responseIdx], LR.predict(scaledModel, X)

            r2Score = Statistics.R2(trainingY, predictionY)
            q2Score = Statistics.Q2(X, trainingY)
            
            iterationHistory[iterationIndex] = (combinations, r2Score, q2Score)
            iterationIndex+=1

            combinations = self.removeLeastSignificantCombination(combinations, scaledModel.conf_int())

        return iterationHistory

    def filterForBestCombinationSet(self, iterationHistory):
        scoreIndex = 2
        getScoreOfSet = lambda setItem: setItem[1][scoreIndex]
        maxScore = getScoreOfSet(max(iterationHistory.items(), key=getScoreOfSet))
        filteredScoreHistory = dict(filter(lambda e: getScoreOfSet(e) > (.95*maxScore if maxScore > 0 else 1.05*maxScore), iterationHistory.items()))
        return min(filteredScoreHistory.items(), key=lambda a: len(a[1][0]))

    def onCall(self):

        combinations = self.getInitCombinations()

        iterationHistory = self.stepwiseRemoveCombinations(combinations)
        
        selctedIndex, (combinations, r2Score, q2Score) = self.filterForBestCombinationSet(iterationHistory)

        scaledModel, model = self.createModels(combinations)
        context.factorSet.setExperimentValueCombinations(combinations)

        context.history.append([selctedIndex, combinations, r2Score, q2Score, [a[1] for a in iterationHistory.values()]])
        
        if len(context.history) >= 10: return StopDoE()

        X = Common.getXWithCombinations(context.experimentValues, combinations, Statistics.orthogonalScaling)

        Common.subplot(
            lambda fig: Statistics.plotScoreHistory([a[1] for a in iterationHistory.values()], selctedIndex, figure=fig),
            lambda fig: Statistics.plotScoreHistory([a[2] for a in iterationHistory.values()], selctedIndex, score="Q2", figure=fig),
            lambda fig: Statistics.plotScoreHistory([a[1] * a[2] for a in iterationHistory.values()], selctedIndex, score="R2Q2", figure=fig),
            lambda fig: Statistics.plotCoefficients(scaledModel.params, context.factorSet, scaledModel.conf_int(), figure=fig),
            lambda fig: Statistics.plotObservedVsPredicted(LR.predict(scaledModel, Common.getXWithCombinations(context.experimentValues, combinations, Statistics.orthogonalScaling)), context.Y[:, 1], X=X, figure=fig),
            lambda fig: Statistics.plotResiduals(Statistics.residualsDeletedStudentized(scaledModel), figure=fig)
        )

        Logger.logEntireRun(len(context.history), context.factorSet, context.experimentValues, context.Y, model.params, scaledModel.params)
        
        return HandleOutliers()


class StopDoE(State):
    def __init__(self): super().__init__("Stop DoE")

    def onCall(self):

        r2ScoreHistory = [e[2] for e in context.history]
        selctedIndex = [e[0] for e in context.history]

        x = list(range(len(context.history)))
        z = np.array([e[4] for e in context.history])

        gP = lambda plt, idx: plt.plot(range(len(z[idx, :])), idx*np.ones(len(z[idx, :])), z[idx, :])

        Common.plot(lambda plt: plt.plot(r2ScoreHistory))
        Common.plot(
            lambda plt: plt.plot(selctedIndex, range(len(context.history)), r2ScoreHistory, 'ro'),
            lambda plt: gP(plt, 0), lambda plt: gP(plt, 1), lambda plt: gP(plt, 2),
            lambda plt: gP(plt, 3), lambda plt: gP(plt, 4), lambda plt: gP(plt, 5),
            lambda plt: gP(plt, 6), lambda plt: gP(plt, 7), lambda plt: gP(plt, 8), 
            lambda plt: gP(plt, 9),
            lambda plt: plt.view_init(azim=0, elev=0),
            is3D=True,
        )
        


class HandleOutliers(State):
    def __init__(self): super().__init__("Handle outliers")

    def detectOutliers(self):
        outlierIdx = context.scaledModel.outlier_test()[:, 0] > 4
        return outlierIdx

    def countExperimentsInRange(self, outlierIdx, range = .05):
        return len(self.getExperimentsInRange(outlierIdx, range))

    def getExperimentsInRange(self, outlierIdx, range = .05):
        X = Statistics.orthogonalScaling(context.experimentValues)
        diff = np.sqrt(((X - X[outlierIdx,:])**2).sum(axis=1))
        return context.experimentValues[diff < range]

    def forEachOutlier(self):
        for idx, isOultier in enumerate(self.detectOutliers()):
            if isOultier: yield (idx, context.experimentValues[idx, :])

    def executeNewExperimentsAroundOutlier(self, outlier):
        newExperimentValues = np.array([outlier]) #np.array([outlier, outlier])
        return np.array([x.getValueArray() for x in context.xamControl.workOffExperiments(newExperimentValues)]), newExperimentValues
  

    def onCall(self):

        if not any(self.detectOutliers()): 
            Logger.logStateInfo("No outliers detected")
            return FindNewExperiments()

        return FindNewExperiments()

        for (idx, outlier) in self.forEachOutlier():
                
            similarExperimentCount = self.countExperimentsInRange(idx)
            Logger.logStateInfo("For Outlier #{} ({}) there are/is {} similar experiment(s)".format(idx, str(outlier), similarExperimentCount-1))

            if similarExperimentCount <= 2:
                repeatedY, newExperimentValues = self.executeNewExperimentsAroundOutlier(outlier)
                repeatLimit = .1

                if all((abs(context.Y[idx, :] - repeatedY) < repeatLimit).reshape(-1, 1)):
                    # All the same again:
                    Logger.logStateInfo("For Outlier #{} ({}) one more measurements resulted in the same outcome -> Remove both".format(idx, str(outlier)))
                    context.deleteExperiment(idx)

                else:
                    # new result:
                    Logger.logStateInfo("For Outlier #{} ({}) one more measurements resulted in a different outcome -> Replace them".format(idx, str(outlier)))
                    context.deleteExperiment(idx)
                    context.addNewExperiments(newExperimentValues, np.array([repeatedY]))

                return EvaluateExperiments()

            else:
                Logger.logStateInfo("Outlier within serveral experiments -> remove all")
                context.deleteExperiment(idx)
                return EvaluateExperiments()


