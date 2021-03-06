from typing import List
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler, PowerTransformer


def getSuggestedTransformer(data):
    # Use, if possible, Box-Cox Transformer
    # otherwise use Yeo-Johnson
    for possibleTransformer in [
            BoxCoxOffsetTransformer(),
        ]:
        
        if possibleTransformer.dataIsValid(data): return possibleTransformer

    return LinearTransformer()

class Transformer:

    def __init__(self, name) -> None:
        self.name = name
        self.transformer = None
        self.fitted = False

    def __str__(self) -> str:
        return "Transformer: {}".format(self.name)

    def transform(self, data, checkData=True):
        if self.transformer is None: raise Exception("Ups 0.o - I am not a real transformer")
        
        if checkData and not self.dataIsValid(data): 
            print("Data is invalid :/ - NO TRANSFORMATION CONDUCTED")
            return data

        data = data.reshape(-1, 1)

        data = self._dataPreTransformation(data)

        self.transformer.fit(data)
        self.fitted = True

        return self.transformer.fit_transform(data)

    def invTransform(self, transformedData):
        if self.transformer is None: raise Exception("Ups 0.o - I am not a real transformer")
        
        if not self.fitted:
            print("Transformer is not fitted :0 - NO (INV) TRANSFORMATION CONDUCTED")
            return transformedData

        data = self.transformer.inverse_transform(transformedData.reshape(-1, 1))
        return self._dataPostInvTransformation(data)

    def dataIsValid(self, data):
        # do any check
        return True

    def _dataPreTransformation(self, data):
        return data

    def _dataPostInvTransformation(self, data):
        return data


class LinearTransformer(Transformer):
    def __init__(self) -> None:
        super().__init__("Linear (NO)")
        self.transformer = self.DummyPowerTransformer()

    class DummyPowerTransformer():
        def __init__(self): pass
        def fit(self, data): return True
        def inverse_transform(self, data): return data
        def fit_transform(self, data): return data


class YeoJohnsonTransformer(Transformer):
    def __init__(self) -> None:
        super().__init__("Yeo-Johnson")
        self.transformer = PowerTransformer(method='yeo-johnson', standardize=True)


class BoxCoxTransformer(Transformer):
    def __init__(self) -> None:
        super().__init__("Box-Cox")
        self.transformer = PowerTransformer(method='box-cox', standardize=True)

    def dataIsValid(self, data):
        return np.all(data > 0)

class BoxCoxOffsetTransformer(Transformer):
    def __init__(self) -> None:
        super().__init__("Box-Cox with Offset")
        self.transformer = PowerTransformer(method='box-cox', standardize=True)
        self.offset = None

    def _dataPreTransformation(self, data):
        dataMin = np.min(data)

        if self.offset is None:
            self.offset = 0 if dataMin > 0 else np.abs(dataMin) + 1e-3

        return self.offset + data

    def _dataPostInvTransformation(self, data):
        return data - self.offset

class LogTransformer(Transformer):
    def __init__(self) -> None:
        super().__init__("Log")
        self.transformer = PowerTransformer(method='box-cox', standardize=True)
        self.c1 = .5
        self.c2 = 1

    def transform(self, data, checkData=True):
        return 10*np.log10(self.c1 + self.c2*data)

    def invTransform(self, transformedData):
        return (10**(transformedData / 10) - self.c1) / self.c2


class NegativeLogTransformer(Transformer):
    def __init__(self) -> None:
        super().__init__("Negative Log")
        self.c1 = 100
        self.c2 = 1

    def transform(self, data, checkData=True):
        return -10*np.log10(self.c1 - self.c2*data)

    def invTransform(self, transformedData):
        return (self.c1 - 10**(transformedData / -10)) / self.c2

class LogitTransformer(Transformer):
    def __init__(self) -> None:
        super().__init__("Logit")
        self.c1 = 0
        self.c2 = 100

    def transform(self, data, checkData=True):
        return 10*np.log10((data - self.c1) / (self.c2 - data))

    def invTransform(self, transformedData):
        A =  10**(transformedData / 10)
        return (A*self.c2 + self.c1) / (1 + A)


class PipeTransformer(Transformer):
    def __init__(self, transformerPipe:List = []) -> None:
        self.transformerPipe = transformerPipe
        super().__init__("Pipe: " + ", ".join([str(tr) for tr in transformerPipe]))

    def transform(self, data, checkData=True):
        for transformer in self.transformerPipe:
            data = transformer.transform(data, checkData)

        return data

    def invTransform(self, transformedData):
        for transformer in self.transformerPipe[::-1]:
            transformedData = transformer.transform(transformedData)

        return transformedData

if __name__ == "__main__":
    TestRun1Data = np.array([0.000539177, 0.066285834, 0.523382715, 0.124081704, 0.340612845, 0.342071572, 0.066361663, 0.414149606, 0.801074925, 0.317247699, 0.137450315, 0.39538596, 0.324086264, 0.360507817, 0.018517327, 0.119053332, 0.569830021, 0.10577096, 0.508014395, 0.174797307, 0.618870595, 0.15085711, 0.144623276, 0.505340362, 0.335538103, 0.341930795, 1.34986653333749])

    for (funcName, func) in {
            #'exp': lambda data: np.exp(data),
            #'log': lambda data: np.log(np.abs(data)),
            #'x^2': lambda data: np.power(data, 2),
            'TestRun#1': lambda data: TestRun1Data
        }.items():

        for transformer in [
                #YeoJohnsonTransformer(),
                #BoxCoxTransformer(),
                BoxCoxOffsetTransformer(),
                LogTransformer(),
                #NegativeLogTransformer(),
                #NegativeLogTransformer(),
                #LogitTransformer()
            ]:

            data = np.random.randn(1000)
            data = func(data)

            tData = transformer.transform(data)
            invTData = transformer.invTransform(tData)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

            ax1.set_title("{} for {}".format(str(transformer), funcName))
            
            ax1.scatter(range(len(data)), data)
            ax1.scatter(range(len(invTData)), invTData)

            ax2.hist(data)
            ax2.hist(invTData)

            ax3.scatter(range(len(tData)), tData)
            ax4.hist(tData)


            plt.show()

    
    
