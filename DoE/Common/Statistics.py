from typing import Callable, Iterable

from matplotlib.pyplot import legend, title
from Common import Common
from Common import Logger
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
import statsmodels.api as sm

from Common.Factor import FactorSet

def plotObservedVsPredicted(prediction, observation, titleSuffix=None, X=None, figure=None):

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
        lambda plt: plt.text(.5*(minVal + maxVal), 0.2, "R2: {}".format(R2(observation, prediction))),
        lambda plt: X is not None and plt.text(.5*(minVal + maxVal), 0.1, "Q2: {}".format(Q2(X, observation))),
        xLabel="Predicted", yLabel="Observed", title=titleStr, 
        figure=figure
    )


def plotResiduals(residuals, bound=4, figure=None):
    rng = range(len(residuals))
    outlierIdx = abs(residuals) > bound

    Common.plot(
        lambda plt: plt.scatter(rng, residuals),
        lambda plt: plt.scatter(np.array(list(rng))[outlierIdx], residuals[outlierIdx], color='red'),
        lambda plt: plt.plot([0, len(residuals)], residuals.mean()*np.array([1, 1]), 'r--'),
        lambda plt: plt.plot([0, len(residuals)], residuals.mean()+bound*np.array([1, 1]), 'k--'),
        lambda plt: plt.plot([0, len(residuals)], residuals.mean()-1*bound*np.array([1, 1]), 'k--'),
        lambda plt: plt.xticks(rng, rng),
        figure=figure
    )


def plotCoefficients(coefficientValues, factorSet:FactorSet=None, confidenceInterval=None, titleSuffix=None, figure=None):
    titleStr = "Coefficients plot"
    if titleSuffix is not None: titleStr += " - " + titleSuffix
    l = len(coefficientValues)
    
    if confidenceInterval is None: 
        confidenceInterval = np.zeros(l)
        isSignificant = np.ones((1, len(l)), dtype=bool)
    else:        
        isSignificant = getModelTermSignificance(confidenceInterval)[0]
        confidenceInterval = abs(confidenceInterval[:, 0] - confidenceInterval[:, 1]) / 2


    labels = None 
    if factorSet is not None or l != len(factorSet.getCoefficientLabels()):
        labels = factorSet.getCoefficientLabels()


    def _plotBars(plt):
        bars = plt.bar(range(l), coefficientValues)
        for index, isSig in enumerate(isSignificant):
            if not isSig: bars[index].set_color('r')

    Common.plot(
        lambda plt: _plotBars(plt),
        lambda plt: plt.errorbar(range(l), coefficientValues, confidenceInterval, fmt=' ', color='b'),
        #lambda plt: plt.xticks(range(l), labels, rotation=90),
        xLabel="Coefficient", yLabel="Value", title=titleStr,
        figure=figure
    )


def plotResponseHistogram(Y, titleSuffix=None):
    titleStr = "Histogram"
    if titleSuffix is not None: titleStr += " - " + titleSuffix

    Common.plot(
        lambda plt: plt.hist(Y),
        xLabel="Coefficient", yLabel="Value", title=titleStr
    )


def plotR2ScoreHistory(r2ScoreHistory, selectedIndex=None, q2ScoreHistory=None, figure=False):

    Common.plot(
        lambda p: p.plot(r2ScoreHistory),
        lambda p: selectedIndex is None or p.scatter(selectedIndex, r2ScoreHistory[selectedIndex], color='r'),
        lambda p: q2ScoreHistory is None or p.plot(q2ScoreHistory, color='k'),
        showLegend= q2ScoreHistory is not None,
        xLabel="Iteration", yLabel="R2 score", title="R2 score over removed combinations",
        figure=figure
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


def Q2(X, Y, roundF : Callable = lambda x: round(x, 5)):
    tmpModel = SMWrapper(sm.OLS).fit(X, Y)
    foldPred = KFold(n_splits=5)

    return roundF(np.mean(cross_val_score(tmpModel, X, Y, cv=foldPred, scoring='r2')))


def getModelTermSignificance(confidenceInterval):
    isSignificant = np.sign(confidenceInterval[:, 1] * confidenceInterval[:, 0]) >= 0
    significanceInterval = np.abs(confidenceInterval[:, 1] - confidenceInterval[:, 0])

    return isSignificant, significanceInterval

def addNoise(X : np.array, sigma = None, mu = 0):
    if sigma is None: sigma = X.std()
    return X + np.random.normal(mu, sigma, X.shape)

def addOutlier(y : np.array, forceIndex=None):
    if forceIndex is None: forceIndex = np.random.randint(0, len(y))
    y[forceIndex] *= 2

    return y

def residualsRaw(observedValues : np.array, predictedValues : np.array) -> np.array:
    return observedValues - predictedValues #model.resid

def residualsStandardized(observedValues : np.array, predictedValues : np.array, residualDegreesOfFreedom) -> np.array:
    residuals = residualsRaw(observedValues, predictedValues)
    return residuals / RSD(residuals, residualDegreesOfFreedom) #r.std()

def residualsDeletedStudentized(model) -> np.array:
    return model.outlier_test()[:, 0]

def RSD(residuals : np.array, residualDegreesOfFreedom):
    return np.sqrt((residuals**2).sum() / (residualDegreesOfFreedom - 2))
    #return r.std()

# For Q2 score wrap extimator from statsmodels...
class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant='add')
        return self.results_.predict(X)