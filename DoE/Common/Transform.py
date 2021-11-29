import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PowerTransformer


def getSuggestedTransformer(data):
    # Use, if possible, Box-Cox Transformer
    # otherwise use Yeo-Johnson
    defaultTransfomer = BoxCoxTransformer()
    if defaultTransfomer.dataIsValid(data): return defaultTransfomer

    return YeoJohnsonTransformer()

class Transformer:

    def __init__(self, name) -> None:
        self.name = name
        self.transformer = None
        self.fitted = False

    def __str__(self) -> str:
        return "Transformer: {}".format(self.name)

    def transform(self, data, checkData=True):
        if self.transformer is None: raise Exception("Ups 0.o - I am no real transformer")
        
        if checkData and not self.dataIsValid(data): 
            print("Data is invalid :/ - NO TRANSFORMATION CONDUCTED")
            return data

        data = data.reshape(-1, 1)

        self.transformer.fit(data)
        self.fitted = True

        return self.transformer.fit_transform(data)

    def invTransform(self, transformedData):
        if self.transformer is None: raise Exception("Ups 0.o - I am no real transformer")
        
        if not self.fitted:
            print("Transformer is not fitted :0 - NO (INV) TRANSFORMATION CONDUCTED")
            return data

        return self.transformer.inverse_transform(transformedData.reshape(-1, 1))

    def dataIsValid(self, data):
        # do any check
        return True

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



if __name__ == "__main__":

    for (funcName, func) in {
            'exp': lambda data: np.exp(data),
            'log': lambda data: np.log(np.abs(data)),
            'x^2': lambda data: np.power(data, 2)
        }.items():

        for transformer in [
                YeoJohnsonTransformer(),
                BoxCoxTransformer()
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

    
    
