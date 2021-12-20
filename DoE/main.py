from Common import Logger
from Common import History
from Common import Optimization
from StateMachine import StateMachine
from StateMachine import DoE

import numpy as np

def main():

    Logger.logInfo("Start main DoE")
    mainSM = StateMachine.StateMachine(DoE.InitDoE(returnAllExperimentsAtOnce=True))
    for state in mainSM: pass


    Logger.logInfo("Find optimum")
    optimum = optimization(state.result())
    Logger.logInfo("Optimum @: {}".format(optimum))


    Logger.logInfo("Start DoE around optimum")
    mainSM = StateMachine.StateMachine(DoE.InitDoE(returnAllExperimentsAtOnce=True))
    for state in mainSM: pass



def optimization(result:History.CombiScoreHistoryItem):
    if result is None: return None

    scaledModel = result.scaledModel
    factorSet = result.context.factorSet
    excludedFactors = result.excludedFactors

    optimum = Optimization.optimizeModel(scaledModel, factorSet.getBounds(excludedFactors))
    optimum = optimum[0:len(optimum)-len(factorSet.experimentValueCombinations)]

    reverseOpt = list(optimum[::-1])
    return [factor.center() if index in excludedFactors else reverseOpt.pop() for index, factor in enumerate(factorSet.factors)]


  
if __name__ == '__main__':

    Logger.initLogging()
    Logger.logInfo("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()

    
