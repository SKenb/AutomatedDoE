import pyDOE2 as pyDOE
import numpy as np

from Common import Logger
#import Logger

class ExperimentFactory:

    def __init__(self):
        self.requestCount = 0

    def getNewExperimentSuggestion(self, factorCount):

        self.requestCount+=1
        experiments = pyDOE.ff2n(factorCount)

        lb, ub = int(2**(factorCount/2)), int(2**(factorCount - 1))
        centerPoint = np.zeros(factorCount)

        #1 Edges and mirrored ones
        if self.requestCount <= 1:
            # Fractional FacDesign
            experiments = experiments[0:lb, :]
            experiments = np.vstack((experiments, -1*experiments))
            return np.vstack((experiments, np.array([centerPoint, centerPoint])))

        elif self.requestCount <= 2:
            # Full FacDesign
            experiments = experiments[lb:ub, :]
            experiments = np.vstack((experiments, -1*experiments))
            return np.vstack((experiments, np.array([centerPoint])))

        elif self.requestCount <= 10:
            # Centered Face Points
            experiments = np.zeros((1, factorCount))
            experiments[0][(self.requestCount - 3) % factorCount] = 1
            experiments = np.vstack((experiments, -1*experiments))
            return experiments

        else:
            Logger.logStateInfo("[WARN] No more experiments :/")
            self.requestCount = 0

            return experiments


if __name__ == "__main__":
    f = ExperimentFactory()
    print(f.getNewExperimentSuggestion())
    print(f.getNewExperimentSuggestion())
    print(f.getNewExperimentSuggestion())
    print(f.getNewExperimentSuggestion())
    print(f.getNewExperimentSuggestion())
    print(f.getNewExperimentSuggestion())
    print(f.getNewExperimentSuggestion())