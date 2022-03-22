from matplotlib.pyplot import ylabel
from matplotlib.transforms import Transform
from StateMachine.StateMachine import State
from StateMachine.Context import ContextDoE
from Common import Common
from Common import Logger
from Common import Statistics
from Common import History
from Common import Paper
from Common import CombinationFactory
from Common import LinearRegression as LR

from scipy.stats import skewtest, boxcox, yeojohnson
from sklearn.preprocessing import quantile_transform

import numpy as np
from pathlib import Path

context = None
history = None

class InitDoE(State):
    def __init__(self, setFactorSet=None, optimum=None, optimumRange=10, returnAllExperimentsAtOnce=False, setXAMControl=None, previousResult=None, previousContext=None): 
        super().__init__("Initialize DoE")

        global context
        context = ContextDoE(setFactorSet, optimum, optimumRange, returnAllExperimentsAtOnce, setXAMControl, previousResult)

        global history
        history = History.History()

        if previousContext is not None:
            context.addNewExperiments(previousContext._experimentValues, previousContext.Y)

            if context.canPredict():
                predictedY = context.predictResponse(previousContext._experimentValues)
                context.addNewPredictedResponses(predictedY)


    def onCall(self):

        Logger.logStateInfo(str(context.factorSet))
        return FindNewExperiments()


class FindNewExperiments(State):
    def __init__(self): super().__init__("Find new experiments")
    def onCall(self):

        experiments = context.experimentFactory.getNewExperimentSuggestion(len(context.factorSet), returnAllExperiments=context.returnAllExperimentsAtOnce)
        if experiments is None: return StopDoE("No more experiments available")

        context.newExperimentValues = context.factorSet.realizeExperiments(experiments, sortColumn=0, sortReverse=len(history) % 2)

        return ExecuteExperiments()

class ExecuteExperiments(State):
    def __init__(self): super().__init__("Execute experiments")
    def onCall(self):


        if len(context.newExperimentValues) <= 0: 
            return EvaluateExperiments() 

        experiment = context.newExperimentValues[0]
        context.newExperimentValues = context.newExperimentValues[1:]
        
        Y = np.array([context.xamControl.startExperimentFromvalues(context.factorSet, experiment).getValueArray()])

        experiment = np.array([experiment])
        context.addNewExperiments(experiment, Y)

        if context.canPredict():
            predictedY = context.predictResponse(experiment)
            measuredY = context.getResponse()[-len(predictedY):]

            context.addNewPredictedResponses(predictedY)

            Logger.logInfo("Measured:   {}".format(measuredY))
            Logger.logInfo("Predicted:  {}".format(predictedY))
            Logger.logInfo("Difference: {}".format((predictedY - measuredY)))

            assert context.getResponse().shape == context.predictedResponses.shape, "Upps"

        return ExecuteExperiments()


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

    def stepwiseRemoveCombinations(self, combinations) -> History.History:
        combiScoreHistory = History.History()

        iterationIndex = 0
        while iterationIndex < 100: #Do not get stuck in loop

            scaledModel, model = self.createModels(combinations)

            X = Common.getXWithCombinations(context.getExperimentValues(), combinations, Statistics.orthogonalScaling)

            trainingY = context.getResponse()
            predictionY = LR.predict(scaledModel, X)
            idxOfCenterPoints = np.linalg.norm(X, axis=1) <= 1e-6

            r2Score = Statistics.R2(trainingY, predictionY)
            q2Score = Statistics.Q2(X, trainingY)
               
            combiScoreHistory.add(History.CombiScoreHistoryItem(
                iterationIndex, combinations, model, 
                scaledModel, context, r2Score, q2Score, context.excludedFactors, 
                {
                    "repScore": Statistics.reproducibility(trainingY[idxOfCenterPoints], trainingY),
                    "CV": Statistics.coefficientOfVariation(trainingY[idxOfCenterPoints])
                })
            )
            
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
        
        history.add(History.DoEHistoryItem(-1, combiScoreHistory, bestCombiScoreItem))
        
        if len(history) >= 40: return StopDoE("Exp. Iteration reached maximum")

        X = Common.getXWithCombinations(context.getExperimentValues(), combinations, Statistics.orthogonalScaling)

        if True:
            Common.subplot(
                lambda fig: Statistics.plotScoreHistory(
                    {
                        "R2": combiScoreHistory.choose(lambda i: i.r2), 
                        "Q2": combiScoreHistory.choose(lambda i: i.q2)
                    }, bestCombiScoreItem.index, figure=fig),
                lambda fig: Statistics.plotResiduals(Statistics.residualsDeletedStudentized(scaledModel), figure=fig),
                lambda fig: Statistics.plotCoefficients(scaledModel.params, context, scaledModel.conf_int(), combinations=combinations, figure=fig),
                #lambda fig: Statistics.plotResponseHistogram(context.getResponse(), figure=fig),
                lambda fig: Statistics.plotObservedVsPredicted(LR.predict(scaledModel, X), context.getResponse(), X=X, figure=fig),
                lambda fig: Statistics.plotResponseHistogram(context.getResponse(), titleSuffix="Response", figure=fig),
                saveFigure=True, title=f"{len(history)}", showPlot=False
            )

            Statistics.plotContour(scaledModel, context.factorSet, context.excludedFactors, combinations, "Plot_C_Iter{}.png".format(len(history)))

        # Paper
        Paper.generatePlot4(
            LR.predict(scaledModel, X), context, scaledModel, combinations, 
            combiScoreHistory, bestCombiScoreItem, drawTicks=False, useLabels=True,
            filename="Plot4_{}_{}_{}Exp.png".format("_Rob" if context.hasOptimum() else "", len(history), len(context._experimentValues))
        )

        Logger.logEntireRun(
            history, context.factorSet, context.excludedFactors, context.getExperimentValues(), context.Y, 
            model.params, scaledModel, context.transformer,
            max(combiScoreHistory.choose(lambda i: i.r2)),
            max(combiScoreHistory.choose(lambda i: i.q2)), 
            max(history.choose(lambda item: item.bestCombiScoreItem.scoreCombis["repScore"])), 
            max(history.choose(lambda item: item.bestCombiScoreItem.scoreCombis["CV"])) 
        )

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

        ## Predicted vs. Measured
        if context.canPredict():
            predictedY, measuredY = context.predictedResponses, context.getResponse()

            print(predictedY.shape)
            print(measuredY.shape)

            error = np.array(predictedY-measuredY)
            Common.subplot(
                lambda fig: Statistics.plotObservedVsPredicted(predictedY, measuredY, "Robustness", suppressR2=True, figure=fig),
                lambda fig: Statistics.plotResiduals(error, bound=False, figure=fig),
                title="Robustness Test",
                saveFigure=True
            )

            Logger.logInfo("Predicted vs. Measured: std: {}".format(np.round(np.std(error), 2)))

        ## Experiments Factor/Response Hist.
        plotter = lambda i: lambda fig: Common.plot(lambda plt: 
                        plt.scatter(list(range(len(context._experimentValues[:, i]))), 
                        context._experimentValues[:, i]), 
                        title=context.factorSet[i], 
                        xLabel="Experiment", yLabel="Value", 
                        figure=fig
                    )
                
        Common.subplot(
            plotter(0), plotter(1), plotter(2), 
            plotter(3),
            saveFigure=True, title="Exp_History"
        )

        #indexTemperature = 3
        #Common.plot(
        #    lambda plt: plt.plot(list(range(len(context._experimentValues[:, indexTemperature]))), context._experimentValues[:, indexTemperature]),
        #    lambda plt: plt.scatter(list(range(len(context._experimentValues[:, indexTemperature]))), context._experimentValues[:, indexTemperature]),
        #    saveFigure=True, title="Temperature"
        #)

        responseStr = ["Space-time yield"]
        plotter = lambda i: lambda fig: Common.plot(
            lambda plt: plt.scatter(list(range(len(context.Y[:, i]))), context.Y[:, i]), 
            title=responseStr[i], 
            xLabel="Experiment", yLabel="Value",
            figure=fig
        )

        Common.subplot(
            plotter(0), 
            saveFigure=True, title="Resp_History"
        )


        ## Stats
        r2ScoreHistory = history.choose(lambda item: item.bestCombiScoreItem.r2)
        q2ScoreHistory = history.choose(lambda item: item.bestCombiScoreItem.q2)
        repScoreHistory = history.choose(lambda item: item.bestCombiScoreItem.scoreCombis["repScore"])
        coefficientOfVariationHistory = history.choose(lambda item: item.bestCombiScoreItem.scoreCombis["CV"])
        selctedIndex = history.choose(lambda item: item.bestCombiScoreItem.index)

        bestScoreOverall = len(q2ScoreHistory) - np.argmax(q2ScoreHistory[::-1]) - 1 #Reverse
        self.bestCombiScoreItemOverall = history.choose(lambda item: item.bestCombiScoreItem)[bestScoreOverall]


        z = lambda pred: np.array(history.choose(lambda item: item.combiScoreHistory.choose(pred)))

        predR2 = lambda item: item.r2
        predQ2 = lambda item: item.q2
        gP = lambda plt, idx, pred: plt.plot(range(len(z(pred)[idx])), idx*np.ones(len(z(pred)[idx])), z(pred)[idx])



        Common.subplot(
            lambda fig: Common.plot(
                            lambda plt: plt.plot(r2ScoreHistory, label="R2"),
                            lambda plt: plt.plot(q2ScoreHistory, label="Q2"),
                            lambda plt: plt.plot(repScoreHistory, label="Reproducibility"),
                            lambda plt: plt.plot(coefficientOfVariationHistory, label="CV"),
                            xLabel="Exp. Iteration", yLabel="Score", title="Score over Exp.It.",
                            showLegend=True, figure=fig
                        ),
            lambda fig: Statistics.plotCoefficients(
                            self.bestCombiScoreItemOverall.scaledModel.params, 
                            self.bestCombiScoreItemOverall.context, 
                            self.bestCombiScoreItemOverall.scaledModel.conf_int(), 
                            combinations=self.bestCombiScoreItemOverall.combinations, 
                            figure=fig
                        ),
            saveFigure=True, title="Score_Best"
        )

        plot3DHist = lambda fig, pred, scoreHistory, title: Common.plot(
            lambda plt: plt.plot(selctedIndex, range(len(history)), scoreHistory, 'ro'),
            lambda plt: gP(plt, 0, pred), lambda plt: gP(plt, 1, pred), lambda plt: gP(plt, 2, pred),
            lambda plt: gP(plt, 3, pred), lambda plt: gP(plt, 4, pred), lambda plt: gP(plt, 5, pred),
            is3D=False, title=title, figure=fig
        )

        Common.subplot(
            lambda fig: plot3DHist(fig, predR2, r2ScoreHistory, "R2 History"),
            lambda fig: plot3DHist(fig, predQ2, q2ScoreHistory, "Q2 Hsitory"),
            is3D=True, saveFigure=True
        )

        # Paper
        title = "Titel Rob" if context.hasOptimum() else "Titel"
        X = Common.getXWithCombinations(context.getExperimentValues(), self.bestCombiScoreItemOverall.combinations, Statistics.orthogonalScaling)

        Paper.generatePlot2(
                LR.predict(self.bestCombiScoreItemOverall.scaledModel, X), 
                context.getResponse(), 
                title, 
                useLabels=True,
                drawOrigin=False,
                drawTicks=True,
                filename=title
            )

        Paper.generatePlot1Bottom(r2ScoreHistory, q2ScoreHistory)
        Paper.generatePlot1(Statistics.orthogonalScaling(context._experimentValues), context.factorSet, filename="ExpHist.png")

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

        return FindNewExperiments()

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


