from typing import Callable, List
from scipy import optimize as opt

import numpy as np

from Common import Logger
from Common import LinearRegression as LR

def optimizeModel(model, bounds, combinations):
    def internPredictor(X):
        X = np.append(X, np.array([f(X) for f in list(combinations.values())]))
        return -1*LR.predict(model, [X])

    return optimize(internPredictor, bounds)

def optimize(eggholder:Callable, bounds:List):

    result = opt.shgo(eggholder, bounds, sampling_method='sobol') #, options={"maxtime": 1e-3})

    if not result.success:
        Logger.logError("OPTIMZATION failed due to: {}".format(result.message))
        return None

    Logger.logInfo("OPTIMZATION -> optimum found @: {}".format(result.x))
    return result.x, -1*result.fun

if __name__ == "__main__":

    testFunc = lambda x: x[1]+x[0]
    optimize(testFunc, [(-1, 1), (-1, 1)])