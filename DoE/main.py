from Common import Common
from Common import Factor
from Common import ExperimentFactory

import numpy as np

import logging

    

def main():
    

    suggestedExperiments = ExperimentFactory.ExperimentFactory()
    factorSet = Factor.getDefaultFactorSet()

    for index, experiments in enumerate(suggestedExperiments):

        print(" A new set of experiments ({}) ".format(index).center(80, '-'))
        print(factorSet.realizeExperiments(experiments))


if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")

    main()
    