import pyDOE2 as pyDOE
import numpy as np

from Common import Logger

class ExperimentFactory:

    def __init__(self):
        self.requestCount = 0

    def getNewExperimentSuggestion(self, factorCount=4):

        self.requestCount+=1
        experiments = pyDOE.ff2n(factorCount)

        lb, ub = int(2**(factorCount/2)), int(2**(factorCount - 1))

        #1 Edges and mirrored ones
        if self.requestCount <= 1:
            experiments = experiments[0:lb, :]
            experiments = np.vstack((experiments, -1*experiments))
            return np.vstack((experiments, np.array([[0, 0, 0, 0]])))

        elif self.requestCount <= 2:
            experiments = experiments[lb:ub, :]
            experiments = np.vstack((experiments, -1*experiments))
            return np.vstack((experiments, np.array([[0, 0, 0, 0], [0, 0, 0, 0]])))

        elif self.requestCount <= 3:
            experiments = experiments[4:2**(factorCount-1), :]
            experiments = np.vstack((experiments, -1*experiments))
            return np.vstack((experiments, np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])))

        else:
            Logger.logStateInfo("[WARN] No more experiments :/")
            self.requestCount = 2
     
            experiments = experiments[0:4, :]
            experiments = np.vstack((experiments, -1*experiments))

            return experiments


if __name__ == "__main__":
    print(ExperimentFactory().getNewExperimentSuggestion())