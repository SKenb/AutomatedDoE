from Common import Logger
from Common import Optimization
from StateMachine import StateMachine
from StateMachine import DoE

import numpy as np

def main():

    Logger.logInfo("Start main DoE")
    mainSM = StateMachine.StateMachine(DoE.InitDoE(returnAllExperimentsAtOnce=True))
    for state in mainSM: pass


    Logger.logInfo("Find optimum")
    optimum = Optimization.optimizationFromDoEResult(state.result())
    Logger.logInfo("Optimum @: {}".format(optimum))


    Logger.logInfo("Start DoE around optimum")
    Logger.appendToLogFolder("DoE_Around_Optimum")
    mainSM = StateMachine.StateMachine(DoE.InitDoE(optimum=optimum, returnAllExperimentsAtOnce=True))
    for state in mainSM: pass


  
if __name__ == '__main__':

    Logger.initLogging()
    Logger.logInfo("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()

    
