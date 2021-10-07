import pyDOE2 as pyDOE
import numpy as np

class ExperimentFactory:

    def __init__(self):
        pass

    def getNewExperimentSuggestion(self, factorCount=4):

        #return pyDOE.ff2n(factorCount)
        b = pyDOE.ff2n(factorCount)
        c = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        return np.vstack((b, c))

if __name__ == "__main__":
    print(ExperimentFactory().getNewExperimentSuggestion())