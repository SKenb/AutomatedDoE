from StateMachine.StateMachine import State
from StateMachine.Context import contextDoE
from Common import Common
from Common import Statistics
from Common import CombinationFactory

import numpy as np

import logging

context = None


class initDoE(State):
    def __init__(self):
        super().__init__("Initialize DoE")
        
    def onCall(self):
        print(self.name)

        global context

        context = contextDoE()
        logging.info(str(context.factorSet))

        return executeExperiments()


class executeExperiments(State):
    def __init__(self):
        super().__init__("Execute Experiments")

    def onCall(self):
        print(self.name)

        experiments = context.experimentFactory.getNewExperimentSuggestion()
        experimentValues = context.factorSet.realizeExperiments(experiments, sortColumn=0)

        context.X = experimentValues
        context.Y = np.array([x.getValueArray() for x in context.xamControl.workOffExperiments(experimentValues)])

        return evaluateExperiments()


class evaluateExperiments(State):
    def __init__(self):
        super().__init__("Evaluate Experiments")

    def onCall(self):
        print(self.name)

        for combinations in [
                            #None,
                            #{"T*R": lambda eV: eV[0]*eV[2], "T*Rt": lambda eV: eV[0]*eV[3]},
                            CombinationFactory.allLinearCombinations()
                        ]:

            context.factorSet.setExperimentValueCombinations(combinations)
            sModel, _ = Common.fitFactorSet(context.factorSet, context.Y[:, 0], verbose=False)
            #Statistics.plotCoefficients(sModel.params, factorSet, sModel.conf_int())

            ## Remove not significant terms
            context.factorSet.removeExperimentValueCombinations(lambda index, key, value: not Statistics.getModelTermSignificance(sModel.conf_int())[0][index])
            sModel, _ = Common.fitFactorSet(context.factorSet, context.Y[:, 0])

        return stopDoE()


class stopDoE(State):
    def __init__(self):
        super().__init__("Stop DoE")

    def onCall(self):
        print(self.name)
        return None
