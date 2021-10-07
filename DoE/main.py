from Common import Common
from Common import Factor
from Common import ExperimentFactory
from Common import CombinationFactory
from Common import LinearRegression
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
    Y = mockY
    
    #Statistics.plotResponseHistogram(Y[:, 0], "Y")

    responseIndexMap = {"Conversion": 0, "Sty": 1}
    response = "Conversion"
    
    for combinations in [
                            None,
                            CombinationFactory.allCombinations(),
                            #{"T*R": lambda eV: eV[0]*eV[2], "T*Rt": lambda eV: eV[0]*eV[3]}
                        ]:
        print(combinations)

        factorSet.setExperimentValueCombinations(combinations)
        print(factorSet)

        X = factorSet.getExperimentValuesAndCombinations()

        #scaledX = factorSet.getExperimentValuesAndCombinations(Statistics.orthogonalScaling)
        #scaledX = Statistics.orthogonalScaling(X)
        #scaledY = Statistics.orthogonalScaling(Y)


        X=sm.add_constant(X)
        model = LinearRegression.fit(X, Y[:, responseIndexMap[response]])
        #sModel = LinearRegression.fit(scaledX, Y[:, responseIndexMap[response]])
        print(model.summary())

        Statistics.plotObservedVsPredicted(model.predict(X), Y[:, responseIndexMap[response]], response)
        #Statistics.plotObservedVsPredicted(sModel.predict(scaledX), scaledY[:, responseIndexMap[response]], response)

        Statistics.plotCoefficients(model.params, factorSet, model.conf_int(alpha=0.05))
        Statistics.test(X, Y[:, responseIndexMap[response]])
        

if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()
    
