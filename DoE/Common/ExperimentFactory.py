import pyDOE2 as pyDOE
import numpy as np

class ExperimentFactory:

    def __init__(self):
        pass

    def getNewExperimentSuggestion(self, factorCount=4):

        return pyDOE.ff2n(factorCount)