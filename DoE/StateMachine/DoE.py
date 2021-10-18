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
        print(self.name)

        global context

        context = contextDoE()
        logging.info(str(context.factorSet))

        return FindNewExperiments()


class FindNewExperiments(State):
    def __init__(self):
        super().__init__("Find new experiments")

    def onCall(self):
        print(self.name)

        experiments = context.experimentFactory.getNewExperimentSuggestion()
        context.newExperimentValues = context.factorSet.realizeExperiments(experiments, sortColumn=0)

        return ExecuteExperiments()

class ExecuteExperiments(State):
    def __init__(self):
        super().__init__("Execute experiments")

    def onCall(self):
        print(self.name)

        Y = np.array([x.getValueArray() for x in context.xamControl.workOffExperiments(context.newExperimentValues)])
        context.addNewExperiment(context.newExperimentValues, Y)

        return EvaluateExperiments()


class EvaluateExperiments(State):
    def __init__(self):
        super().__init__("Evaluate experiments")

    def onCall(self):
        print(self.name)

        for combinations in [
                            #None,
                            #{"T*R": lambda eV: eV[0]*eV[2], "T*Rt": lambda eV: eV[0]*eV[3]},
                            CombinationFactory.allLinearCombinations()
                        ]:

            X = Common.getXWithCombinations(context.experimentValues, combinations)
            scaledX = Common.getXWithCombinations(context.experimentValues, combinations, Statistics.orthogonalScaling)
            context.model = LR.fit(X, context.Y[:, 0])
            context.scaledModel = LR.fit(scaledX, context.Y[:, 0])

            # Remove non significant
            combinations = Common.removeCombinations(combinations, lambda index, key, value: not Statistics.getModelTermSignificance(context.scaledModel.conf_int())[0][index])
            
            context.model = Common.getModel(context.experimentValues, combinations, context.Y[:, 0])
            context.scaledModel = Common.getModel(context.experimentValues, combinations, context.Y[:, 0], Statistics.orthogonalScaling)
    
            Statistics.plotObservedVsPredicted(LR.predict(context.scaledModel, Common.getXWithCombinations(context.experimentValues, combinations, Statistics.orthogonalScaling)), context.Y[:, 0], X=scaledX)
            Statistics.plotResiduals(Statistics.residualsDeletedStudentized(context.scaledModel))

        return HandleOutliers()


class StopDoE(State):
    def __init__(self):
        super().__init__("Stop DoE")

    def onCall(self):
        print(self.name)
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
        print(self.name)

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


