from Common import Common
from Common import Factor
from Common import ExperimentFactory
from XamControl import XamControl

import numpy as np
import logging

def main():
    
    xamControl = XamControl.XamControlSimpleMock()
    experimentFactory = ExperimentFactory.ExperimentFactory()
    factorSet = Factor.getDefaultFactorSet()
    logging.info(str(factorSet))

    experiments = experimentFactory.getNewExperimentSuggestion()
    experimentValues = factorSet.realizeExperiments(experiments, sortColumn=0)

    X = np.array([x.getFactorResponseArray() for x in xamControl.workOffExperiments(experimentValues)])
    print(X)


if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()
    