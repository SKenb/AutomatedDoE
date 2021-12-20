from typing import Callable, List
from scipy import optimize as opt

import numpy as np

from Common import Logger
from Common import History
from Common import LinearRegression as LR

def optimizationFromDoEResult(result:History.CombiScoreHistoryItem):
    if result is None: return None

    model = result.model
    factorSet = result.context.factorSet
    excludedFactors = result.excludedFactors

    optimum = optimizeModel(model, factorSet.getBounds(excludedFactors))
    optimum = optimum[0:len(optimum)-len(factorSet.experimentValueCombinations)]

    reverseOpt = list(optimum[::-1])
    return [factor.center() if index in excludedFactors else reverseOpt.pop() for index, factor in enumerate(factorSet.factors)]

def optimizeModel(model, bounds):
    return optimize(lambda X: LR.predict(model, [X]), bounds)

def optimize(eggholder:Callable, bounds:List):

    result = opt.shgo(eggholder, bounds, sampling_method='sobol')

    if not result.success:
        Logger.logError("OPTIMZATION failed due to: {}".format(result.message))
        return None

    Logger.logInfo("OPTIMZATION -> optimum found @: {}".format(result.x))
    return result.x

if __name__ == "__main__":

    testFunc = lambda x: x[1]+x[0]
    optimize(testFunc, [(-1, 1), (-1, 1)])