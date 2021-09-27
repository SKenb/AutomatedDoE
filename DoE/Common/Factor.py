
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


class FactorSet:

    def __init__(self, factors=[]):
        self.factors = factors

    def addFactor(self, factor):
        self.factors.append(factor)

    def __str__(self):
        return "Factor Set:\n\r\t" + "\n\r\t".join(map(str, self.factors))


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