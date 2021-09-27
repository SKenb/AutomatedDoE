import pyDOE2 as pyDOE
import numpy as np

class ExperimentFactory():

    def __init__(self, factorCount=4, maxIterations=10):
        self.maxIterations = maxIterations
        self.factorCount = factorCount

    def __iter__(self):
        self.iterationIndex = 0
        return self

    def __next__(self):
        self.iterationIndex+=1

        if self.iterationIndex <= self.maxIterations:
            
            # Return a set of Experiments
            return pyDOE.bbdesign(self.factorCount)

        else:
            raise StopIteration