from Common import Logger
from Common import History
from Common import Optimization
from XamControl import XamControl
from StateMachine import StateMachine
from StateMachine import DoE

import numpy as np

def main():

    Logger.logInfo("Start main DoE")
    mainSM = StateMachine.StateMachine(DoE.InitDoE(setXAMControl=XamControl.XamControlTestRun1Mock()))
    for state in mainSM: pass

    Logger.logInfo("Find optimum")
    optimum = optimization(state.result())
    Logger.logInfo("Optimum @: {}".format(optimum))

    Logger.logInfo("Start DoE around optimum")
    Logger.appendToLogFolder("DoE_Around_Optimum")
    mainSM = StateMachine.StateMachine(DoE.InitDoE(optimum=optimum))
    for state in mainSM: pass



def optimization(result:History.CombiScoreHistoryItem):
    if result is None: return None

    model = result.model
    factorSet = result.context.factorSet
    excludedFactors = result.excludedFactors

    bounds = factorSet.getBounds(excludedFactors)
    red = lambda i: [fB[0] for fB in bounds]

    combiBounds = [(func(red(0)), func(red(1))) for func in result.combinations.values()]

    bounds.extend(combiBounds)
    optimum = Optimization.optimizeModel(model, bounds)
    optimum = optimum[0:len(optimum)-len(factorSet.experimentValueCombinations)]

    reverseOpt = list(optimum[::-1])
    return [factor.center() if index in excludedFactors else reverseOpt.pop() for index, factor in enumerate(factorSet.factors)]


  
if __name__ == '__main__':

    Logger.initLogging()
    Logger.logInfo("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()

    
