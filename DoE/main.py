from Common import Common
from Common import Factor
from Common import ExperimentFactory
from XamControl import XamControlExperiment
from XamControl import XamControl

import numpy as np

import logging

def main():
    
    xamControl = XamControl.XamControlMock()
    experimentFactory = ExperimentFactory.ExperimentFactory()
    factorSet = Factor.getDefaultFactorSet()

    logging.info(str(factorSet))

    experiments = experimentFactory.getNewExperimentSuggestion()
    experimentValues = factorSet.realizeExperiments(experiments, sortColumn=0)
    print(experimentValues)

    result = xamControl.doExperiment(XamControlExperiment.createExperimentRequestFromValues(experimentValues))

    print(result)


if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")

    main()
    