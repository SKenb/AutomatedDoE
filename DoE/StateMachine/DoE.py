from StateMachine.StateMachine import State
from StateMachine.Context import ContextDoE
from Common import Common
from Common import Logger
from Common import Statistics
from Common import History
from Common import CombinationFactory
from Common import LinearRegression as LR

from scipy.stats import skewtest, boxcox, yeojohnson
from sklearn.preprocessing import quantile_transform

import numpy as np

context = None


class InitDoE(State):
    def __init__(self, optimum=None, optimumRange=10, returnAllExperimentsAtOnce=False): 
        super().__init__("Initialize DoE")

        self.returnAllExperimentsAtOnce = returnAllExperimentsAtOnce
        self.optimum = optimum
        self.optimumRange = optimumRange
        
    def onCall(self):

        global context

        context = ContextDoE(self.optimum, self.optimumRange, self.returnAllExperimentsAtOnce)
        Logger.logStateInfo(str(context.factorSet))

        return FindNewExperiments()


class FindNewExperiments(State):
    def __init__(self): super().__init__("Find new experiments")
    def onCall(self):

        experiments = context.experimentFactory.getNewExperimentSuggestion(len(context.factorSet), returnAllExperiments=context.returnAllExperimentsAtOnce)
        if experiments is None: return StopDoE("No more experiments available")

        context.newExperimentValues = context.factorSet.realizeExperiments(experiments, sortColumn=3, sortReverse=len(context.history) % 2)

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
        return CombinationFactory.allCombinations(context.activeFactorCount()) # CombinationFactory.allLinearCombinations(context.activeFactorCount()) # 

    def createModels(self, combinations, responseIdx = 1):

        context.scaledModel = Common.getModel(context.getExperimentValues(), combinations, context.getResponse(), Statistics.orthogonalScaling)
        context.model = Common.getModel(context.getExperimentValues(), combinations, context.getResponse())

        return context.scaledModel, context.model

    def removeLeastSignificantFactorOrCombination(self, combinations, model):
        isSignificant, significanceInterval = Statistics.getModelTermSignificance(model.conf_int())

        significanceInterval[np.abs(model.params) < 1e-4] = 1000 # we remove factors / combinations which are near zero
        significanceInterval[isSignificant] = 0 # we won't remove significant
        significanceInterval[0] = 0 # we won't remove constant part

        if(len(significanceInterval) != 1 + context.activeFactorCount() + len(combinations)):
            raise Exception("Ups")
        
        minIndex = [index for index, value in enumerate(significanceInterval) if value >= max(significanceInterval)][-1] #np.argmax(significanceInterval)
        isFactor = minIndex-1 < context.activeFactorCount()

        if isFactor:
            # we need to remove a factor ->
            # before check if any combination with this factor still exists
            factorIndex = context.getFactorSetIndexFromCoefIndex(minIndex-1)
            idx = Common.combinationIndexContainingFactor(combinations, factorIndex)

            if len(idx) <= 0:
                #No combinations with factor -> remove factor
                context.excludeFactor(factorIndex)
                combinations = CombinationFactory.removeFactor(combinations, factorIndex, self.getInitCombinations())
            else:
                #Last combination first
                lastCombinationWithFactor = idx[-1]
                combinations = Common.removeCombinations(combinations, lambda index, k, v: index == lastCombinationWithFactor, -1) 
        else:
            #Remove least significant
            combinations = Common.removeCombinations(combinations, lambda index, k, v: index == (minIndex - context.activeFactorCount() - 1), -1) 

        return combinations 

    def getCombinationsForBinPattern(self, combinationSet, number):
        removeList = list(range(5))
        removeList.extend([int(d) for d in str(bin(number))[2:]])
        removeList.extend(np.zeros(len(combinationSet)))

        return Common.removeCombinations(combinationSet, lambda index, k, v: removeList[index] > 0) 

    def stepwiseRemoveCombinations(self, combinations, responseIdx = 1) -> History.History:
        combiScoreHistory = History.History()

        iterationIndex = 0
        while iterationIndex < 100: #Do not get stuck in loop

            scaledModel, model = self.createModels(combinations)

            X = Common.getXWithCombinations(context.getExperimentValues(), combinations, Statistics.orthogonalScaling)

            trainingY = context.getResponse()
            predictionY = LR.predict(scaledModel, X)

            r2Score = Statistics.R2(trainingY, predictionY)
            q2Score = Statistics.Q2(X, trainingY)

            # Used as different scores so far
            scoreCombis = {
                "R2*Q2": r2Score*q2Score, 
                "1-(R2-Q2)": (1-(r2Score-q2Score))
            }
            
            combiScoreHistory.add(History.CombiScoreHistoryItem(iterationIndex, combinations, model, scaledModel, context, r2Score, q2Score, context.excludedFactors, scoreCombis))
            
            isSignificant, _ = Statistics.getModelTermSignificance(scaledModel.conf_int())

            if len(combinations) <= 0 and all(isSignificant): break
            if len(combinations) <= 0 and len(scaledModel.params) <= 1: break


            combinations = self.removeLeastSignificantFactorOrCombination(combinations, scaledModel)
            iterationIndex = iterationIndex+1

        return combiScoreHistory

    def filterForBestCombinationSet(self, combiScoreHistory : History.History) -> History.CombiScoreHistoryItem:

        valueOfInterest = lambda item: item.q2 # scoreCombis["1-(R2-Q2)"] #item.r2 # item.q2 # 
        search = lambda func: valueOfInterest(func(combiScoreHistory.items(), key=valueOfInterest))
        maxScore = search(max)

        if np.isnan(maxScore): 
            # Scores results are not usefull
            return combiScoreHistory[-1]

        relScoreBound = (maxScore - search(min)) * 0.05
        bound = (maxScore-relScoreBound) if maxScore > 0 else (maxScore+relScoreBound)

        filteredCombiScoreHistory = combiScoreHistory.filter(lambda item: valueOfInterest(item) >= bound)

        return min(filteredCombiScoreHistory, key=lambda item: len(item.combinations)-len(item.excludedFactors))

    def onCall(self):

        context.resetFactorExlusion()
        combinations = self.getInitCombinations()

        combiScoreHistory = self.stepwiseRemoveCombinations(combinations)
        bestCombiScoreItem = self.filterForBestCombinationSet(combiScoreHistory)
        
        combinations = bestCombiScoreItem.combinations
        
        context.resetFactorExlusion()
        context.excludeFactor(bestCombiScoreItem.excludedFactors)
        context.factorSet.setExperimentValueCombinations(combinations)

        scaledModel, model = self.createModels(combinations)
        
        context.history.add(History.DoEHistoryItem(-1, combiScoreHistory, bestCombiScoreItem))
        
        if len(context.history) >= 40: return StopDoE("Exp. Iteration reached maximum")

        X = Common.getXWithCombinations(context.getExperimentValues(), combinations, Statistics.orthogonalScaling)

        combis = list(combiScoreHistory[0].scoreCombis.keys())


        if True:
            Common.subplot(
                lambda fig: Statistics.plotScoreHistory(
                    {
                        "R2": combiScoreHistory.choose(lambda i: i.r2), 
                        "Q2": combiScoreHistory.choose(lambda i: i.q2),
                        combis[0]: combiScoreHistory.choose(lambda i: i.scoreCombis[combis[0]]),
                        combis[1]: combiScoreHistory.choose(lambda i: i.scoreCombis[combis[1]])
                    }, bestCombiScoreItem.index, figure=fig),
                lambda fig: Statistics.plotResiduals(Statistics.residualsDeletedStudentized(scaledModel), figure=fig),
                lambda fig: Statistics.plotCoefficients(scaledModel.params, context, scaledModel.conf_int(), combinations=combinations, figure=fig),
                #lambda fig: Statistics.plotResponseHistogram(context.getResponse(), figure=fig),
                lambda fig: Statistics.plotObservedVsPredicted(LR.predict(scaledModel, X), context.getResponse(), X=X, figure=fig),
                lambda fig: Statistics.plotResponseHistogram(context.getResponse(), titleSuffix="Response", figure=fig),
                saveFigure=True, title=f"{len(context.history)}", showPlot=False
            )

        Logger.logEntireRun(context.history, context.factorSet, context.getExperimentValues(), context.Y, model.params, scaledModel.params)

        return HandleOutliers()


class StopDoE(State):
    def __init__(self, reason): 
        super().__init__("Stop DoE")
        self.stopReason = reason
        self.bestCombiScoreItemOverall = None

    def result(self):
        return self.bestCombiScoreItemOverall

    def onCall(self):

        Logger.logInfo("STOP due to: {}".format(self.stopReason))

        ## Experiments Factor/Response Hist.
        plotter = lambda i: lambda fig: Common.plot(lambda plt: plt.scatter(list(range(len(context._experimentValues[:, i]))), context._experimentValues[:, i]), title=context.factorSet[i], figure=fig)
        Common.subplot(
            plotter(0), plotter(1), plotter(2), 
            plotter(3), plotter(4), plotter(5), 
            saveFigure=True, title="Exp_History"
        )

        indexTemperature = 3
        Common.plot(
            lambda plt: plt.plot(list(range(len(context._experimentValues[:, indexTemperature]))), context._experimentValues[:, indexTemperature]),
            lambda plt: plt.scatter(list(range(len(context._experimentValues[:, indexTemperature]))), context._experimentValues[:, indexTemperature]),
            saveFigure=True, title="Temperature"
        )

        responseStr = ["Space-time yield"]
        plotter = lambda i: lambda fig: Common.plot(lambda plt: plt.scatter(list(range(len(context.Y[:, i]))), context.Y[:, i]), title=responseStr[i], figure=fig)
        Common.subplot(
            plotter(0), 
            saveFigure=True, title="Resp_History"
        )


        ## Stats

        r2ScoreHistory = context.history.choose(lambda item: item.bestCombiScoreItem.r2)
        q2ScoreHistory = context.history.choose(lambda item: item.bestCombiScoreItem.q2)
        combiScoreHistory = context.history.choose(lambda item: item.bestCombiScoreItem.scoreCombis["1-(R2-Q2)"])
        selctedIndex = context.history.choose(lambda item: item.bestCombiScoreItem.index)

        bestScoreOverall = len(q2ScoreHistory) - np.argmax(q2ScoreHistory[::-1]) - 1 #Reverse
        self.bestCombiScoreItemOverall = context.history.choose(lambda item: item.bestCombiScoreItem)[bestScoreOverall]


        z = lambda pred: np.array(context.history.choose(lambda item: item.combiScoreHistory.choose(pred)))

        predR2 = lambda item: item.r2
        predQ2 = lambda item: item.q2

        gP = lambda plt, idx, pred: plt.plot(range(len(z(pred)[idx])), idx*np.ones(len(z(pred)[idx])), z(pred)[idx])
        #plotRO = lambda yValues: lambda plt: plt.plot(bestCombiScoreItemOverall.index, yValues[bestCombiScoreItemOverall.index], 'ro')

        Common.subplot(
            lambda fig: Common.plot(
                            lambda plt: plt.plot(r2ScoreHistory, label="R2"),
                            lambda plt: plt.plot(q2ScoreHistory, label="Q2"),
                            lambda plt: plt.plot(context.history.choose(lambda item: item.bestCombiScoreItem.scoreCombis["1-(R2-Q2)"]), label="1-(R2-Q2)"),
                            #plotRO(r2ScoreHistory), plotRO(q2ScoreHistory),
                            showLegend=True, figure=fig
                        ),
            lambda fig: Statistics.plotCoefficients(
                            self.bestCombiScoreItemOverall.scaledModel.params, 
                            self.bestCombiScoreItemOverall.context, 
                            self.bestCombiScoreItemOverall.scaledModel.conf_int(), 
                            combinations=self.bestCombiScoreItemOverall.combinations, 
                            figure=fig
                        ),
            saveFigure=True, title="Score_Best", rows=2, cols=2
        )

        plot3DHist = lambda fig, pred, scoreHistory, title: Common.plot(
            lambda plt: plt.plot(selctedIndex, range(len(context.history)), scoreHistory, 'ro'),
            lambda plt: gP(plt, 0, pred), lambda plt: gP(plt, 1, pred), lambda plt: gP(plt, 2, pred),
            lambda plt: gP(plt, 3, pred), lambda plt: gP(plt, 4, pred), lambda plt: gP(plt, 5, pred),
            lambda plt: gP(plt, 6, pred), lambda plt: gP(plt, 7, pred), lambda plt: gP(plt, 8, pred), 
            lambda plt: gP(plt, 9, pred),
            is3D=False, title=title, figure=fig
        )

        #Common.subplot(
        #    lambda fig: plot3DHist(fig, predR2, r2ScoreHistory, "R2 History"),
        #    lambda fig: plot3DHist(fig, predQ2, q2ScoreHistory, "Q2 Hsitory"),
        #    is3D=True
        #)
        


class HandleOutliers(State):
    def __init__(self): super().__init__("Handle outliers")

    def detectOutliers(self) -> np.array:
        outlierIdx = np.abs(context.scaledModel.outlier_test()[:, 0]) > 4
        return outlierIdx

    def countExperimentsInRange(self, outlierIdx, range = .05):
        return len(self.getExperimentsInRange(outlierIdx, range))

    def getExperimentsInRange(self, outlierIdx, range = .05):
        X = Statistics.orthogonalScaling(context._experimentValues)
        diff = np.sqrt(((X - X[outlierIdx,:])**2).sum(axis=1))
        return context._experimentValues[diff < range]

    def forEachOutlier(self):
        for idx, isOultier in enumerate(self.detectOutliers()):
            if isOultier: yield (idx, context._experimentValues[idx, :])

    def executeNewExperimentsAroundOutlier(self, outlier):
        newExperimentValues = np.array([outlier]) #np.array([outlier, outlier])
        return np.array([x.getValueArray() for x in context.xamControl.workOffExperiments(newExperimentValues)]), newExperimentValues
  

    def onCall(self):

        context.restoreDeletedExperiments()

        if not any(self.detectOutliers()): 
            Logger.logStateInfo("No outliers detected")
            return FindNewExperiments()

        if self.detectOutliers().sum() > 3: 
            Logger.logStateInfo("Too many outliers detected - Maybe bad intermediate model")
            return FindNewExperiments()

        #return FindNewExperiments()

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


