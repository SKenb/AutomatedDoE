from Common import Common
from Common import Factor
from Common import ExperimentFactory
from Common import CombinationFactory
from Common import LinearRegression as LR
from Common import Statistics
from XamControl import XamControl

import statsmodels.api as sm
import numpy as np
import logging

def main():
    
    xamControl = XamControl.XamControlSimpleMock() #XamControl.XamControlNoMixtureTermsMock()
    experimentFactory = ExperimentFactory.ExperimentFactory()
    factorSet = Factor.getDefaultFactorSet()
    logging.info(str(factorSet))

    experiments = experimentFactory.getNewExperimentSuggestion()
    experimentValues = factorSet.realizeExperiments(experiments, sortColumn=0)

    mockY = np.array([x.getValueArray() for x in xamControl.workOffExperiments(experimentValues)])
    moddeY = Common.getModdeTestResponse()
    Y = moddeY
    
    #Statistics.plotResponseHistogram(Y[:, 0], "Y")

    responseIndexMap = {"Conversion": 0, "Sty": 1}
    response = "Conversion"
    
    for combinations in [
                            None,
                            {"T*R": lambda eV: eV[0]*eV[2], "T*Rt": lambda eV: eV[0]*eV[3]},
                        ]:
        print(combinations)

        factorSet.setExperimentValueCombinations(combinations)
        print(factorSet)

        X = factorSet.getExperimentValuesAndCombinations()
        scaledX = factorSet.getExperimentValuesAndCombinations(Statistics.orthogonalScaling)

        model = LR.fit(X, Y[:, responseIndexMap[response]])
        sModel = LR.fit(scaledX, Y[:, responseIndexMap[response]])
        print(sModel.summary())

        #Statistics.plotObservedVsPredicted(LR.predict(model, X), Y[:, responseIndexMap[response]], response)
        Statistics.plotObservedVsPredicted(LR.predict(sModel, scaledX), Y[:, responseIndexMap[response]], response)

        Statistics.plotCoefficients(sModel.params, factorSet, sModel.conf_int())
        Statistics.test(X, Y[:, responseIndexMap[response]])
        

if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()
    
