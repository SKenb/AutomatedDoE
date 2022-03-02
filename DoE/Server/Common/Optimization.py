from typing import Callable, List
from scipy import optimize as opt

import time

from Common import Logger
from Common import LinearRegression as LR

def optimizeModel(model, bounds):
    return optimize(lambda X: LR.predict(model, [X]), bounds)

def optimize(eggholder:Callable, bounds:List):

    result = opt.shgo(eggholder, bounds, sampling_method='sobol', options={"maxtime": 1e-3})

    if not result.success:
        Logger.logError("OPTIMZATION failed due to: {}".format(result.message))
        return None

    Logger.logInfo("OPTIMZATION -> optimum found @: {}".format(result.x))
    return result.x

if __name__ == "__main__":

    testFunc = lambda x: x[1]+x[0]
    optimize(testFunc, [(-1, 1), (-1, 1)])