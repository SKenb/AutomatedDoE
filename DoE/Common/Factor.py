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

    def __init__(self, factors=[]):
        self.factors = factors

    def addFactor(self, factor):
        self.factors.append(factor)

    def __str__(self):
        return "Factor Set:\n\r\t" + "\n\r\t".join(map(str, self.factors))

    def realizeExperiments(self, nomredExperiments, sortColumn=None):
        result = np.array([self * experiment for experiment in nomredExperiments])
        
        if sortColumn is not None and sortColumn < self.getFactorCount():
            return result[result[:, sortColumn].argsort()]

        return result

    def __mul__(self, other):
        return [
                factor * other[index] 
                for index, factor in enumerate(self.factors)
            ]

    def __rmul__(self, other):
        self.__mul__(other)

    def getFactorCount(self):
        return len(self.factors)



def getDefaultFactorSet():

    return FactorSet([
        Factor("Temperature", 60, 160, "°C", "T_1"),
        Factor("Concentration", .2, .4, "M", "C_SM_1"),
        Factor("Reagent ratio‘s ", .8, 3, "", "R"),
        Factor("Residence time", 0, 100, "min", "RT_1"),
    ])

if __name__ == '__main__':
    
    print((" Test Factor class ".center(80, "-")))

    print(getDefaultFactorSet())

    print(getDefaultFactorSet()*np.array([-1, -1, 0, 1]))

    print(getDefaultFactorSet().realizeExperiments(np.array([
            [-1, -1, 0, 0], 
            [0, -1, 0, 1],
            [0, -1, 1, 0]
        ])))
