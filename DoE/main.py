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
    
    xamControl = XamControl.XamControlSimpleMock() #XamControl.XamControlNoMixtureTermsMock()
    experimentFactory = ExperimentFactory.ExperimentFactory()
    factorSet = Factor.getDefaultFactorSet()
    logging.info(str(factorSet))

    experiments = experimentFactory.getNewExperimentSuggestion()
    experimentValues = factorSet.realizeExperiments(experiments, sortColumn=0)

    mockY = np.array([x.getValueArray() for x in xamControl.workOffExperiments(experimentValues)])
    mockY = Statistics.addNoise(mockY, .05)

    moddeY = Common.getModdeTestResponse()
    
    Y = moddeY
    
    #Statistics.plotResponseHistogram(Y[:, 0], "Y")

    responseIndexMap = {"Conversion": 0, "Sty": 1}
    response = "Conversion"
    
    for combinations in [
                            None,
                            #{"T*R": lambda eV: eV[0]*eV[2]},
                            #CombinationFactory.allLinearCombinations()
                        ]:

        factorSet.setExperimentValueCombinations(combinations)
        sModel, _ = Common.fitFactorSet(factorSet, Y[:, responseIndexMap[response]])

        exit()
        X = factorSet.getExperimentValuesAndCombinations()
        tmp(X, 0.9, 0.2, 2.5, 60)
        tmp(X, 0.9, 0.2, 6, 60)
        tmp(X, 0.9, 0.4, 2.5, 60)
        tmp(X, 0.9, 0.4, 6, 60)
        tmp(X, 3, 0.2, 2.5, 60)
        tmp(X, 3, 0.2, 6, 60)
        tmp(X, 3, 0.4, 2.5, 60)
        tmp(X, 3, 0.4, 6, 60)
        tmp(X, 1.95, 0.3, 4.25, 110)
        tmp(X, 1.95, 0.3, 4.25, 110)
        tmp(X, 1.95, 0.3, 4.25, 110)
        tmp(X, 0.9, 0.2, 2.5, 160)
        tmp(X, 0.9, 0.2, 6, 160)
        tmp(X, 0.9, 0.4, 2.5, 160)
        tmp(X, 0.9, 0.4, 6, 160)
        tmp(X, 3, 0.2, 2.5, 160)
        tmp(X, 3, 0.2, 6, 160)
        tmp(X, 3, 0.4, 2.5, 160)
        tmp(X, 3, 0.4, 6, 160)

        ## Remove not significant terms
        #factorSet.removeExperimentValueCombinations(lambda index, key, value: not Statistics.getModelTermSignificance(sModel.conf_int())[0][index])
        #sModel, _ = Common.fitFactorSet(factorSet, Y[:, responseIndexMap[response]])
    
def tmp(X, ratio, conc, resT, temp):
    ws = np.append(X, Common.getModdeTestResponse(), axis=1)
    a = ws[ws[:, 0] == temp]
    a = a[a[:, 1] == conc]
    a = a[a[:, 2] == ratio]
    a = a[a[:, 3] == resT][0, :]
    print(">> {}\t{}\t{}\t{}\t>> {}\t{}".format(
        a[2], a[1], a[3], a[0], 
        a[4], a[5]
    ))
    return a[4:6]

if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")
    np.set_printoptions(suppress=True)
    
    main()
    
