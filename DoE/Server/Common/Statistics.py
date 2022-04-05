from typing import Callable, Dict, Iterable

from matplotlib.pyplot import title, xlabel, ylabel
from Common import Common
from Common.Factor import FactorSet, getDefaultFactorSet
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin
from Common import LinearRegression as LR
from matplotlib import cm


import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from StateMachine.Context import ContextDoE

def plotObservedVsPredicted(prediction, observation, titleSuffix=None, X=None, figure=None, suppressR2=False, savePath=None):
    titleStr = "Observed vs. Predicted"
    if titleSuffix is not None: titleStr += " - " + titleSuffix

    red = lambda func: func(func(prediction), func(observation))

    minVal = red(min)
    maxVal = red(max)

    Common.plot(
        lambda plt: plt.scatter(prediction, observation),
        lambda plt: plt.plot([minVal, maxVal], [minVal, minVal], 'k', linewidth=1),
        lambda plt: plt.plot([minVal, minVal], [minVal, maxVal], 'k', linewidth=1),
        lambda plt: plt.plot([minVal, maxVal], [minVal, maxVal], 'k--', linewidth=2),
        lambda plt: plt.grid(), 
        #lambda plt: True if suppressR2 else plt.text(.05*(maxVal - minVal), -.15*(maxVal - minVal), "R2: {}".format(round(R2(observation, prediction), 2))),
        #lambda plt: X is not None and plt.text(.4*(maxVal - minVal), -.15*(maxVal - minVal), "Q2: {}".format(round(Q2(X, observation), 2))),
        xLabel=r"Predicted STY ($kg\;L^{-1}\;h^{-1}$)", yLabel=r"Observed STY ($kg\;L^{-1}\;h^{-1}$)", title=titleStr, 
        saveFigure=savePath is not None,
        savePath=savePath,
        figure=figure
    )


def plotResiduals(residuals, bound=4, figure=None):
    rng = range(len(residuals))
    outlierIdx = abs(residuals) > bound

    Common.plot(
        lambda plt: plt.scatter(rng, residuals),
        lambda plt: plt.scatter(np.array(list(rng))[outlierIdx], residuals[outlierIdx], color='red'),
        lambda plt: plt.plot([0, len(residuals)], residuals.mean()*np.array([1, 1]), 'r--'),
        lambda plt: True if bound is None else plt.plot([0, len(residuals)], residuals.mean()+bound*np.array([1, 1]), 'k--'),
        lambda plt: True if bound is None else plt.plot([0, len(residuals)], residuals.mean()-1*bound*np.array([1, 1]), 'k--'),
        #lambda plt: plt.text(2, 1.2*residuals.mean(), "Std: {} ({}%)".format(
        #        np.round(np.std(residuals), 4), 
        #        np.round(np.std(residuals) / residuals.mean() * 100, 2)
        #    )),
        #lambda plt: plt.xticks(rng, rng),
        title="Residuals", xLabel="Experiment", yLabel="Residual",
        figure=figure
    )


def plotCoefficients(coefficientValues, context:ContextDoE=None, confidenceInterval=None, titleSuffix=None, figure=None, combinations:dict=None):
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
    if context is not None and combinations is not None and l == context.activeFactorCount() + len(combinations) + 1:
        char = lambda index: chr(65 + index % 26)

        #labels = ["Constant"]
        #labels.extend(["{} ({})".format(context.factorSet[index], char(index)) for index in range(len(context.factorSet)) if not context.isFactorExcluded(index)])
        labels = ["0"]
        labels.extend(["{}".format(char(index)) for index in range(len(context.factorSet)) if not context.isFactorExcluded(index)])
        labels.extend(["${}$".format(key.replace("*", " \cdot ")) for key in combinations.keys()])


    def _plotBars(plt):
        bars = plt.bar(range(l), coefficientValues)
        for index, isSig in enumerate(isSignificant):
            if not isSig: bars[index].set_color('r')

    Common.plot(
        lambda plt: _plotBars(plt),
        lambda plt: plt.errorbar(range(l), coefficientValues, confidenceInterval, fmt=' ', color='b'),
        lambda plt: True if labels is None else plt.xticks(range(l), labels, rotation=90),
        xLabel="Coefficient", yLabel="Magnitude", title=titleStr,
        figure=figure
    )


def plotResponseHistogram(Y, titleSuffix=None, figure=None):
    titleStr = "Histogram"
    if titleSuffix is not None: titleStr += " - " + titleSuffix

    Common.plot(
        lambda plt: plt.hist(Y),
        xLabel="Response", yLabel="Value", title=titleStr,
        figure=figure
    )


def plotScoreHistory(scoreHistoryDict : Dict, selectedIndex=None, figure=False):

    def plotAllScores(p):
        for _, (score, scoreHistory) in enumerate(scoreHistoryDict.items()):
            p.plot(scoreHistory, label="${}$".format(score.replace("2", "^2")))
            
            if selectedIndex is not None:
                p.scatter(selectedIndex, scoreHistory[selectedIndex], color='r'),
 
    Common.plot(
        plotAllScores,
        showLegend= len(scoreHistoryDict) > 1,
        xLabel="Iteration", yLabel="Score", 
        #title=("" if len(scoreHistoryDict) > 1 else scoreHistoryDict[0].keys()[0]) + "Score",
        title=r"$R^2$ and $Q^2$",
        figure=figure
    )

def plotContour2(model : sm.OLS, factorSet : FactorSet, excludedFactors, combinations, filename=None, figure=None):

    plt.rcParams.update({
        'text.usetex': True,
        'font.size': '18',
        'font.weight': 'bold'
    })

    delta = 0.005

    indexX, indexY = 0, 3

    x = np.arange(factorSet.factors[indexX].min, factorSet.factors[indexX].max+delta, delta)
    y = np.arange(factorSet.factors[indexY].min, factorSet.factors[indexY].max+delta, delta)
    X, Y = np.meshgrid(x, y)

    x1, y1 = X.reshape(-1), Y.reshape(-1)

    maxIndizes = [2, 4]
    responses = np.array([factor.center() if index not in maxIndizes else factor.max for index, factor in enumerate(factorSet.factors)])
    responses = np.array([responses.T for i in range(len(x1))])

    responses[:, indexX] = x1
    responses[:, indexY] = y1

    responses = np.delete(responses,  excludedFactors, axis=1) 

    RX = [np.append(r, [f(r) for f in combinations.values()]) for r in responses]

    Z = LR.predict(model, RX)

    zMin = min(Z)
    zMax = max(Z)

    levels = np.linspace(zMin, zMax, 10)

    Common.plot(
        lambda fig: fig.contourf(X, Y, Z.reshape(X.shape), levels, cmap=cm.RdYlGn),
        lambda fig: fig.xticks([factorSet.factors[indexX].min, factorSet.factors[indexX].center(), factorSet.factors[indexX].max]),
        lambda fig: fig.yticks([factorSet.factors[indexY].min, factorSet.factors[indexY].center(), factorSet.factors[indexY].max]),
        xLabel=factorSet[indexX], yLabel=factorSet[indexY], title="",
        saveFigure=filename is not None, 
        setFilename=filename, figure=figure
    )



def plotContour(model : sm.OLS, factorSet : FactorSet, excludedFactors, combinations, filename=None):
    delta = 0.025

    indexX, indexY, indexZ = 0, 3, 4

    x = np.arange(factorSet.factors[indexX].min, factorSet.factors[indexX].max+delta, delta)
    y = np.arange(factorSet.factors[indexY].min, factorSet.factors[indexY].max+delta, delta)
    X, Y = np.meshgrid(x, y)

    x1, y1 = X.reshape(-1), Y.reshape(-1)

    responses = np.array([factor.center() for factor in factorSet.factors])
    responses = np.array([responses.T for i in range(len(x1))])

    responses[:, indexX] = x1
    responses[:, indexY] = y1

    responses = np.delete(responses,  excludedFactors, axis=1) 

    z, zMin, zMax = [], None, None
    layers = 9
    w = np.linspace(factorSet.factors[indexZ].min, factorSet.factors[indexZ].max, layers)

    for index in range(layers):

        responses[:, indexZ] = w[index]*np.ones(x1.shape)
        RX = [np.append(r, [f(r) for f in combinations.values()]) for r in responses]

        zNew = LR.predict(model, RX)
        z.append(zNew)

        if zMin is None or zMin > min(zNew): zMin = min(zNew)
        if zMax is None or zMax < max(zNew): zMax = max(zNew)

    levels = np.linspace(zMin, zMax, 10)

    def contourSubplot(figure, X, Y, Z, levels, xlabel="", ylabel="", title=""):
        Common.plot(
            lambda fig: fig.contourf(X, Y, Z, levels),
            figure=figure,
            xLabel=xlabel, yLabel=ylabel, title=title
        )

    xlabel = factorSet[indexX]
    ylabel = factorSet[indexY]
    titlesRed = ["{}".format(round(w_, 2)) for w_ in w]
    titles = ["{} = {}".format(factorSet[indexZ], round(w_, 2)) for w_ in w]

    def subplot_(idx, titles, xlabel="", ylabel=""):
        return lambda fig: contourSubplot(fig, X, Y, z[idx].reshape(X.shape), levels, xlabel, ylabel, titles[idx])

    Common.subplot(
        subplot_(0, titlesRed), subplot_(1, titles), subplot_(2, titlesRed),
        subplot_(3, titlesRed, ylabel=ylabel), subplot_(4, titlesRed), subplot_(5, titlesRed),
        subplot_(6, titlesRed), subplot_(7, titlesRed, xlabel=xlabel), subplot_(8, titlesRed),
        title="Contour", saveFigure=True, setFilename=filename
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
    #return roundF(1 - np.mean(np.abs(cross_val_score(SMWrapper(sm.OLS), X, Y, cv=5, scoring="neg_mean_squared_error"))))
    #return roundF(np.mean(cross_val_score(SMWrapper(sm.OLS), X, Y, cv=KFold(shuffle=True, n_splits=5), scoring=meanAbsolutePercentageError)))
    return roundF(np.mean(cross_val_score(SMWrapper(sm.OLS), X, Y, cv=5, scoring="r2").clip(0)))


def meanAbsolutePercentageError(clf, X, y, epsilon = 1e-6):
    yPred = clf.predict(X)

    for index, yi in enumerate(y):
        if abs(yi) < epsilon: y[index] = epsilon
      
    return (1/len(y))*np.sum(np.abs(y - yPred)/y) / 100


def reproducibility(Ypure, Ytotal):
    MSpe, MStot = np.var(Ypure), np.var(Ytotal)
    return (1 - (MSpe / MStot))

def coefficientOfVariation(Y):
    return np.std(Y) / np.mean(Y)


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
            X = sm.add_constant(X, has_constant='add')
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant='add')
        return self.results_.predict(X)