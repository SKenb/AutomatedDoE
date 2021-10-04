from matplotlib.pyplot import ylabel
from Common import Common
from Common import Factor
from Common import ExperimentFactory
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
                            {"Temp*Conc": lambda eV: eV[0] * eV[1], "Temp*ResT": lambda eV: eV[0] * eV[3]},
                            {"Temp*Ratio": lambda eV: eV[0] * eV[2], "Temp*ResT": lambda eV: eV[0] * eV[3]},
                            {"Temp*Conc": lambda eV: eV[0] * eV[1], "Temp*Ratio": lambda eV: eV[0] * eV[2]},
                        ]:

        factorSet.setExperimentValueCombinations(combinations)
        print(factorSet)

        model = LinearRegression.fit(factorSet.getExperimentValuesAndCombinations(), Y[:, 1]) 
        Statistics.plotObservedVsPredicted(model.predict(factorSet.getExperimentValuesAndCombinations()), Y[:, 1], "Conversion")


if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()
    
