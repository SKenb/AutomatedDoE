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

    def delta(self): return self.max - self.min

    def center(self): return self.min + self.delta() / 2

    def __mul__(self, other):
        return round(self.center() + self.delta() / 2 * other, 10)

    def __rmul__(self, other):
        return self.__mul__(other)


class FactorSet:

    def __init__(self, factors : Iterable[Factor]=[]):
        self.factors = factors
        self.realizedExperiments = None
        self.experimentValueCombinations = None

    def addFactor(self, factor : Iterable[Factor]):
        self.factors.append(factor)

    def __str__(self):
        resultStr = "Factor Set:\n\r\t" + "\n\r\t".join(map(str, self.factors))
        if self.experimentValueCombinations is None: return resultStr

        return resultStr + "\n\r\t+ Combis:\n\r\t\t" + "\n\r\t\t".join(map(str, self.experimentValueCombinations.keys()))


    def realizeExperiments(self, nomredExperiments : Iterable, sortColumn=None):
        self.realizedExperiments = nomredExperiments
        self._setExperimentValues(np.array([self * experiment for experiment in nomredExperiments]))
        self.sortExperimentValues(sortColumn)
        return self.getExperimentValues()

    def getRealizedExperiments(self):
        return self.realizedExperiments

    def _setExperimentValues(self, valueArray):
        self.experimentValues = valueArray

    def sortExperimentValues(self, sortColumn):
        if sortColumn is not None and sortColumn < self.getFactorCount():
            self._setExperimentValues(self.experimentValues[self.experimentValues[:, sortColumn].argsort()])

    def getExperimentValues(self):
        return self.experimentValues

    def getExperimentValuesAndCombinations(self, scalingFunction : Callable = None) -> np.array:
        if scalingFunction is None: scalingFunction = lambda x: x
        if self.experimentValueCombinations is None: return scalingFunction(self.getExperimentValues())

        scaledExperimentValues = scalingFunction(self.getExperimentValues())
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

    def __mul__(self, other):
        return [
                factor * other[index] 
                for index, factor in enumerate(self.factors)
            ]

    def __rmul__(self, other):
        self.__mul__(other)

    def getFactorCount(self):
        return len(self.factors)

    def getCoefficientLabels(self):
        labels = ["Intercept"]
        labels.extend([factor.name for factor in self.factors])

        if self.experimentValueCombinations is not None: 
            labels.extend(self.experimentValueCombinations.keys())

        return labels


def getDefaultFactorSet():

    return FactorSet([
        Factor("Temperature", 60, 160, "°C", "T_1"),
        Factor("Concentration", .2, .4, "M", "C_SM_1"),
        Factor("Reagent ratio‘s", .9, 3, "", "R"),
        Factor("Residence time", 2.5, 6, "min", "RT_1"),
    ])

if __name__ == '__main__':
    
    print((" Test Factor class ".center(80, "-")))

    fSet = getDefaultFactorSet()

    print(fSet)

    print(fSet*np.array([-1, -1, 0, 1]))

    print(fSet.realizeExperiments(np.array([
            [-1, -1, 0, 0], 
            [0, -1, 0, 1],
            [0, -1, 1, 0]
        ])))

    fSet.setExperimentValueCombinations({"Temp*A": lambda a: a[0]*a[1]})
    print(fSet)