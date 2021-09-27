from Common import Common
from Common import Factor
from Common import ExperimentFactory

import numpy as np

import logging

    

def main():
    

    suggestedExperiments = ExperimentFactory.ExperimentFactory()
    factorSet = Factor.getDefaultFactorSet()

    logging.info(str(factorSet))

    for index, experiments in enumerate(suggestedExperiments):

        print(" A new set of experiments ({}) ".format(index+1).center(80, '-'))

        experimentValues = factorSet.realizeExperiments(experiments, sortColumn=0)
        print(experimentValues)


if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")

    main()
    