from Common import Common
from StateMachine import StateMachine
from StateMachine import DoE

import numpy as np
import logging

def main():

    mainSM = StateMachine.StateMachine(DoE.InitDoE())
    for _ in mainSM: pass
    
if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()
    
