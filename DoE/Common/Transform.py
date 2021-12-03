import numpy as np
import matplotlib.pyplot as plt

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
        self.offset = 0

    def _dataPreTransformation(self, data):
        dataMin = np.min(data)
        self.offset = 0 if dataMin > 0 else np.abs(dataMin) + 1e-3

        return self.offset + data

    def _dataPostInvTransformation(self, data):
        return data - self.offset




if __name__ == "__main__":

    for (funcName, func) in {
            'exp': lambda data: np.exp(data),
            'log': lambda data: np.log(np.abs(data)),
            'x^2': lambda data: np.power(data, 2)
        }.items():

        for transformer in [
                YeoJohnsonTransformer(),
                BoxCoxTransformer(),
                BoxCoxOffsetTransformer()
            ]:

            data = np.random.randn(1000)
            data = func(data)

            tData = transformer.transform(data)
            invTData = transformer.invTransform(tData)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

            ax1.set_title("{} for {}".format(str(transformer), funcName))
            
            ax1.plot(data)
            ax1.plot(invTData)

            ax2.hist(data)
            ax2.hist(invTData)

            ax3.plot(tData)
            ax4.hist(tData)


            plt.show()

    
    
