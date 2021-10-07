from typing import Callable
from Common import Common
from sklearn.metrics import r2_score
from sklearn import preprocessing
import numpy as np
import statsmodels.api as sm

from Common.Factor import FactorSet

def plotObservedVsPredicted(prediction, observation, titleSuffix=None):

    titleStr = "Observed vs. Predicted"
    if titleSuffix is not None: titleStr += " - " + titleSuffix

    red = lambda func: func(func(prediction), func(observation))

    minVal = red(min)
    maxVal = red(max)

    Common.plot(
        lambda plt: plt.scatter(prediction, observation),
        lambda plt: plt.plot([minVal, maxVal], [0, 0], 'k', linewidth=1),
        lambda plt: plt.plot([0, 0], [minVal, maxVal], 'k', linewidth=1),
        lambda plt: plt.plot([minVal, maxVal], [minVal, maxVal], 'k--', linewidth=2),
        lambda plt: plt.grid(), 
        lambda plt: plt.text(.5*(minVal + maxVal), 0.1, "R2: {}".format(R2(observation, prediction))),
        xLabel="Predicted", yLabel="Observed", title=titleStr
    )


def plotCoefficients(coefficentValues, factorSet:FactorSet=None, confidenceInterval=None, titleSuffix=None):
    titleStr = "Coefficients plot"
    if titleSuffix is not None: titleStr += " - " + titleSuffix
    l = len(coefficentValues)
    
    if confidenceInterval is None: 
        confidenceInterval = np.zeros(l)
    else:
        confidenceInterval = abs(confidenceInterval[:, 0] - confidenceInterval[:, 1]) / 2

    labels = None 
    if factorSet is not None or l != len(factorSet.getCoefficientLabels()):
        labels = factorSet.getCoefficientLabels()

    print(range(l))
    print(labels)

    Common.plot(
        lambda plt: plt.bar(range(l), coefficentValues),
        lambda plt: plt.errorbar(range(l), coefficentValues, confidenceInterval, fmt=' ', color='b'),
        lambda plt: plt.xticks(range(l), labels, rotation=90),
        xLabel="Coefficient", yLabel="Value", title=titleStr
    )


def plotResponseHistogram(Y, titleSuffix=None):
    titleStr = "Histogram"
    if titleSuffix is not None: titleStr += " - " + titleSuffix

    Common.plot(
        lambda plt: plt.hist(Y),
        xLabel="Coefficient", yLabel="Value", title=titleStr
    )


def generateScaler(X):
    return preprocessing.StandardScaler().fit(X)

def scale(X):
    return generateScaler(X).transform(X)

def orthogonalScaling(X, axis=0):
    minX = np.min(X, axis=axis)
    maxX = np.max(X, axis=axis)
    R = (maxX - minX) / 2

    return midRangeScaling(X, axis) / R

def midRangeScaling(X, axis=0):
    minX = np.min(X, axis=axis)
    maxX = np.max(X, axis=axis)
    M = (minX + maxX) / 2

    return (X-M)

def unitVarianceScaling(X, axis=0):
    mean = np.mean(X, axis=axis)
    std = np.std(X, axis=axis)

    return (X - mean) / std

def R2(observation, prediction, roundF : Callable = lambda x: round(x, 5)):
    if roundF is None: roundF = lambda x: x
    return roundF(r2_score(observation, prediction))


def combineCoefficients(model) -> np.array:
    c = [model.intercept_]
    c.extend(model.coef_)
    return np.array(c)


def test(X, yObserved):
    X=sm.add_constant(X)
    model= sm.OLS(yObserved, X).fit()
    predictions= model.summary()
    predictions