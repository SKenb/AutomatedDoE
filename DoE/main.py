from Common import Logger
from StateMachine import StateMachine
from StateMachine import DoE

import numpy as np

def main():

    Logger.logInfo("Start StateMachine with InitDoE")
    mainSM = StateMachine.StateMachine(DoE.InitDoE())
    for _ in mainSM: pass
    
if __name__ == '__main__':

    Logger.initLogging()
    Logger.logInfo("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()
    
