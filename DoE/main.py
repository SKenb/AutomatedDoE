from Common import Common
from Common import Factor
from Common import ExperimentFactory
from Common import CombinationFactory
from Common import LinearRegression
from Common import Statistics
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

    Y = np.array([x.getValueArray() for x in xamControl.workOffExperiments(experimentValues)])
    
    for combinations in [
                            None, 
                            CombinationFactory.allCombinations()
                        ]:
        print(combinations)

        factorSet.setExperimentValueCombinations(combinations)
        print(factorSet)

        model = LinearRegression.fit(factorSet.getExperimentValuesAndCombinations(), Y[:, 1]) 

        Statistics.plotObservedVsPredicted(model.predict(factorSet.getExperimentValuesAndCombinations()), Y[:, 1], "Conversion")

        Common.plot(
            lambda plt: plt.errorbar(range(len(model.coef_)), model.coef_,)
        )


if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()
    
