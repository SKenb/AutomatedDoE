from Common import Common
from Common import Factor
from Common import ExperimentFactory
from Common import CombinationFactory
from Common import Statistics
from XamControl import XamControl

import statsmodels.api as sm
import numpy as np
import logging

def main():
    
    xamControl = XamControl.XamControlModdeYMock() #XamControl.XamControlSimpleMock() #XamControl.XamControlNoMixtureTermsMock()
    experimentFactory = ExperimentFactory.ExperimentFactory()
    factorSet = Factor.getDefaultFactorSet()
    logging.info(str(factorSet))

    experiments = experimentFactory.getNewExperimentSuggestion()
    experimentValues = factorSet.realizeExperiments(experiments, sortColumn=0)

    mockY = np.array([x.getValueArray() for x in xamControl.workOffExperiments(experimentValues)])
    #mockY = Statistics.addNoise(mockY, .05)

    Y = mockY
    print(Y)
    
    #Statistics.plotResponseHistogram(Y[:, 0], "Y")

    responseIndexMap = {"Conversion": 0, "Sty": 1}
    response = "Conversion"
    
    for combinations in [
                            #None,
                            #{"T*R": lambda eV: eV[0]*eV[2], "T*Rt": lambda eV: eV[0]*eV[3]},
                            CombinationFactory.allLinearCombinations()
                        ]:

        factorSet.setExperimentValueCombinations(combinations)
        sModel, _ = Common.fitFactorSet(factorSet, Y[:, responseIndexMap[response]], verbose=False)
        #Statistics.plotCoefficients(sModel.params, factorSet, sModel.conf_int())

        ## Remove not significant terms
        factorSet.removeExperimentValueCombinations(lambda index, key, value: not Statistics.getModelTermSignificance(sModel.conf_int())[0][index])
        sModel, _ = Common.fitFactorSet(factorSet, Y[:, responseIndexMap[response]])
        pass


def tmp(X, ratio, conc, resT, temp):
    ws = np.append(X, Common.getModdeTestResponse(), axis=1)
    print(ws)

if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()
    
