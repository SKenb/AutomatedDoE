from Common import Common
from Common import Factor
from Common import ExperimentFactory
from Common import CombinationFactory
from Common import Statistics
from XamControl import XamControl

from StateMachine import StateMachine
from StateMachine import DoE

import statsmodels.api as sm
import numpy as np
import logging

def main():

    mainSM = StateMachine.StateMachine(DoE.initDoE())
    for _ in mainSM: pass
    
if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()
    
