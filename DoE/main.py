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
    
    model = LinearRegression.fit(experimentValues, Y[:, 1]) 
    print(model.predict([experimentValues[1, :]]))
    print(Y[1, 1])

    factorSet.setExperimentValueCombinations({"Temp*Res": lambda eV: eV[0] * eV[3], "Temp*Ratio": lambda eV: eV[0] * eV[2]})
    model = LinearRegression.fit(factorSet.getExperimentValuesAndCombinations(), Y[:, 1]) 
    predY = model.predict(factorSet.getExperimentValuesAndCombinations())

    Statistics.plotObservedVsPredicted(predY, Y[:, 1], "Conversion")


 


if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()
    