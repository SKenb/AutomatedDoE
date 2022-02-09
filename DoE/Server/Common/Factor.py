from tkinter.ttk import Separator
from typing import Callable, Dict, Iterable
import numpy as np 

class Factor:

    def __init__(self, name, min=0.0, max=1.0, unit='', symbol=''):
        self.name = name
        self.unit = unit
        self.symbol = symbol
        self.min = -1e6
        self.max = -self.min

        self.setMax(max)
        self.setMin(min)


    def __str__(self):
        return "{} {}  >> ({} - {} {})".format(
            self.name, 
            self.symbol,
            self.min, 
            self.max, 
            self.unit
        )

    def setMin(self, minValue): self.min = minValue

    def setMax(self, maxValue): self.max = maxValue

    def getBounds(self): return (self.min, self.max)

    def delta(self): return self.max - self.min

    def center(self): return self.min + self.delta() / 2

    def __mul__(self, other):
        return round(self.center() + self.delta() / 2 * other, 10)

    def __rmul__(self, other):
        return self.__mul__(other)

    def transformToOptimum(self, optimum, range):
        deltaAroundOptimum = self.delta() * range / 100

        self.min = optimum - deltaAroundOptimum/2
        self.max = optimum + deltaAroundOptimum/2

        return self


class FactorSet:

    def __init__(self, factors : Iterable[Factor]=[]):
        self.factors = factors
        self.realizedExperiments = None
        self.experimentValueCombinations = None

    def addFactor(self, factor : Iterable[Factor]):
        self.factors.append(factor)

    def __str__(self):
        resultStr = self.getFactorString()
        if self.experimentValueCombinations is None: return resultStr

        return resultStr + self.getCombinationsString()

    def getFactorString(self, multiLine=True):
        separatorString = "\n\r\t" if multiLine else "\t"
        return "Factor Set:{}".format(separatorString) + separatorString.join(map(str, self.factors))

    def getCombinationsString(self, multiLine=True):
        separatorString = "\n\r\t\t" if multiLine else "\t\t"
        return "{}+ Combis:{}".format(separatorString, separatorString) + separatorString.join(map(str, self.factors))

    def __len__(self):
        return len(self.factors)

    def __mul__(self, other):
        return [
                factor * other[index] 
                for index, factor in enumerate(self.factors)
            ]

    def __rmul__(self, other):
        self.__mul__(other)

    def __getitem__(self, index):
        return self.factors[index].name
    
    def realizeExperiments(self, nomredExperiments : Iterable, sortColumn=None, sortReverse=False):
        self.realizedExperiments = nomredExperiments
        self._setExperimentValues(np.array([self * experiment for experiment in nomredExperiments]))
        self.sortExperimentValues(sortColumn, sortReverse)
        return self.getExperimentValues()

    def getRealizedExperiments(self):
        return self.realizedExperiments

    def _setExperimentValues(self, valueArray):
        self.experimentValues = valueArray

    def sortExperimentValues(self, sortColumn, reverse=False):
        if sortColumn is not None and sortColumn < self.getFactorCount():
            
            idx = self.experimentValues[:, sortColumn].argsort()
            if reverse: idx = idx[::-1]

            self._setExperimentValues(self.experimentValues[idx])

    def getExperimentValues(self):
        return self.experimentValues

    def getExperimentValuesAndCombinations(self, scalingFunction : Callable = None, experimentValues : np.array = None) -> np.array:
        experimentValues = self.getExperimentValues() if experimentValues is None else experimentValues
        if scalingFunction is None: scalingFunction = lambda x: x
        if self.experimentValueCombinations is None: return scalingFunction(experimentValues)

        scaledExperimentValues = scalingFunction(experimentValues)
        # Non scaled combinations
        combinations = np.array([
            np.array([
                    func(e) for func in self.experimentValueCombinations.values()
                ]) for e in scaledExperimentValues
        ])

        # Combination of scaled combinations and factors
        return np.append(scaledExperimentValues, combinations, axis=1)

    def setExperimentValueCombinations(self, newCombinations : Dict):
        self.experimentValueCombinations = newCombinations
    
    def resetExperimentValueCombinations(self):
        self.setExperimentValueCombinations(None)

    def removeExperimentValueCombinations(self, removePredicate : Callable):
        # Remove combinations
        reduced = { key:value for index, (key,value) in  
            enumerate(self.experimentValueCombinations.items()) 
            if not removePredicate(index + len(self.factors) + 1, key, value)
        }
        
        self.setExperimentValueCombinations(reduced)
        return reduced


    def getFactorCount(self):
        return len(self.factors)

    def getCoefficientLabels(self):
        labels = ["Intercept"]
        labels.extend([factor.name for factor in self.factors])

        if self.experimentValueCombinations is not None: 
            labels.extend(self.experimentValueCombinations.keys())

        return labels

    def getBounds(self, excludedFactors):
        normedExperiments = np.array([
            -1*np.ones(self.getFactorCount()),
            np.ones(self.getFactorCount())
        ])

        self.realizeExperiments(normedExperiments)
        return [(rExp[0], rExp[1]) for index, rExp in enumerate(self.getExperimentValuesAndCombinations().T) if index not in excludedFactors]


def getDefaultFactorSet():

    return FactorSet([
        Factor("Temperature", 100, 160, "°C", "T_1"),
        Factor("Concentration", .2, .4, "M", "C_SM_1"),
        Factor("Reagent ratio", .9, 3, "", "R"),
        Factor("Residence time", 2.5, 6, "min", "RT_1"),
        #Factor("Dummy factor 1", -100, 100, "knolls", "DF1"),
        #Factor("Dummy factor 2", -10, 10, "knolls", "DF2"),
    ])

def getFactorSetAroundOptimum(baseFactorSet, optimum, optimumRange=10):
    return FactorSet([factor.transformToOptimum(optimum[index], optimumRange) for index, factor in enumerate(baseFactorSet.factors)])

if __name__ == '__main__':
    
    print((" Test Factor class ".center(80, "-")))

    fSet = getDefaultFactorSet()

    print(fSet)

    print(fSet*np.array([-1, -1, 0, 1, 0]))

    print(fSet.realizeExperiments(np.array([
            [-1, -1, 0, 0, -1], 
            [0, -1, 0, 1, 0],
            [0, -1, 1, 0, 1]
        ])))

    fSet.setExperimentValueCombinations({"Temp*A": lambda a: a[0]*a[1]})
    print(fSet)