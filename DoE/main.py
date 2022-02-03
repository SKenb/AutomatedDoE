from tracemalloc import Statistic
from Common import Logger
from Common import History
from Common import Statistics
from Common import Optimization
from Common.Factor import FactorSet, getDefaultFactorSet
from XamControl import XamControl
from StateMachine import StateMachine
from StateMachine import DoE
import statsmodels.api as sm

import numpy as np

def main():

    Logger.logInfo("Start main DoE")
    mainSM = StateMachine.StateMachine(DoE.InitDoE(setXAMControl=XamControl.XamControlTestRun1Mock()))
    for state in mainSM: pass

    Logger.logInfo("Find optimum")
    optimum = optimization(state.result())
    Logger.logInfo("Optimum @: {}".format(optimum))

    Statistics.plotContour(
        state.result().scaledModel, 
        getDefaultFactorSet(), 
        state.result().excludedFactors, 
        state.result().combinations
    )
    exit()
      
    Logger.logInfo("Start DoE around optimum")
    Logger.appendToLogFolder("DoE_Around_Optimum")
    mainSM = StateMachine.StateMachine(
        DoE.InitDoE(
            optimum=optimum,
            previousResult=state.result(),
            previousContext=state.result().context,
            setXAMControl=XamControl.XamControlTestRun1RobustnessMock()
        )
    )
    for state in mainSM: pass


def optimization(result:History.CombiScoreHistoryItem):
    if result is None: return None

    model = result.model
    factorSet = result.context.factorSet
    excludedFactors = result.excludedFactors

    # do this in loop
    result.context.factorSet.setExperimentValueCombinations(result.combinations)
    
    bounds = factorSet.getBounds(excludedFactors)

    optimum = Optimization.optimizeModel(model, bounds)
    optimum = optimum[0:len(optimum)-len(factorSet.experimentValueCombinations)]

    reverseOpt = list(optimum[::-1])
    return [factor.min if index in excludedFactors else reverseOpt.pop() for index, factor in enumerate(factorSet.factors)]


  
if __name__ == '__main__':

    Logger.initLogging()
    Logger.logInfo("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()

    
