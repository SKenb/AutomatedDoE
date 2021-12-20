import pyDOE2 as pyDOE
import numpy as np

from Common import Logger
#import Logger

class ExperimentFactory:

    def __init__(self):
        self.requestCount = 0

    def getNewExperimentSuggestion(self, factorCount, temperatureColumn = 3, returnAllExperiments=False):

        self.requestCount+=1
        experiments = pyDOE.ff2n(factorCount)
        centerPoint = np.zeros(factorCount)

        if returnAllExperiments:
            if self.requestCount <= 1:
                # Add center points
                experiments = np.vstack((experiments, np.array([centerPoint, centerPoint, centerPoint])))
                # Add face center points
                for index in range(factorCount):
                    faceCenterPoints = np.zeros((1, factorCount))
                    faceCenterPoints[0][int(index % factorCount)] = 1
                    faceCenterPoints = np.vstack((faceCenterPoints, -1*faceCenterPoints))

                    experiments = np.vstack((experiments, faceCenterPoints))

                return experiments
            else:
                Logger.logStateInfo("[WARN] No more experiments :/")
                return None
    
        power = min(np.ceil(factorCount/2), 3)
        edgeExperimentCount = 2**(factorCount-power-1)

        if temperatureColumn != 0:
            # Replace column zero with temperatureColumn --> rising temperature
            experiments[:, [0, temperatureColumn]] = experiments[:, [temperatureColumn, 0]]

        if self.requestCount <= edgeExperimentCount:
            # Edges
            n = self.requestCount-1

            lowerBound = int(n*2**power)
            upperBound = int(((n+1)*2**power) - 1)

            experiments = experiments[lowerBound:upperBound, :]
            experiments = np.vstack((experiments, -1*experiments))

            centerPoints = np.array([centerPoint])
            if self.requestCount <= 1:
                centerPoints = np.array([centerPoint, centerPoint])
            
            return np.vstack((experiments, centerPoints))

        elif self.requestCount <= edgeExperimentCount+factorCount:
            # Centered Face Points
            experiments = np.zeros((1, factorCount))
            experiments[0][int((self.requestCount - edgeExperimentCount - 1 + temperatureColumn) % factorCount)] = 1
            experiments = np.vstack((experiments, -1*experiments))
            return experiments

        else:
            Logger.logStateInfo("[WARN] No more experiments :/")
            #self.requestCount = 0

            return None


if __name__ == "__main__":
    f = ExperimentFactory()
    print(f.getNewExperimentSuggestion())
    print(f.getNewExperimentSuggestion())
    print(f.getNewExperimentSuggestion())
    print(f.getNewExperimentSuggestion())
    print(f.getNewExperimentSuggestion())
    print(f.getNewExperimentSuggestion())
    print(f.getNewExperimentSuggestion())