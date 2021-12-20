from typing import Callable, Iterable
import matplotlib.pyplot as plt
import Common.LinearRegression as LR
import numpy as np
import Common.Statistics as Statistics
import Common.Logger as Logger
import random   

from datetime import datetime
from pathlib import Path

from Common.Factor import FactorSet

import matplotlib

MATPLOTLIB_SAVE_PLOT_ONLY = True


def plot(*plotters, is3D=False, xLabel="x", yLabel="y", title="Plot", showLegend=False, figure=None, saveFigure=False):
     
    if figure is None: 
        figure = plt.figure()
        showPlot = True
    else:
        showPlot = False

    if is3D: ax = figure.add_subplot(111, projection='3d')
    
    for plotter in plotters: plotter(ax if is3D else plt)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)

    if showLegend: plt.legend()
    if showPlot and not MATPLOTLIB_SAVE_PLOT_ONLY: plt.show()
    if MATPLOTLIB_SAVE_PLOT_ONLY: matplotlib.use('Agg')

    if saveFigure: 
        filename = Path("Plot_{}.png".format(title))
        path = Logger.getCurrentLogFolder() / filename
        plt.savefig(path)

    return figure


def subplot(*plotFunctions, is3D=False, saveFigure=False, title="", showPlot=True, rows=None, cols=None):
    
    if rows is None and cols is None:
        cols = np.ceil((np.sqrt(len(plotFunctions))))
        rows = np.ceil(len(plotFunctions) / cols)
    else:
        if rows is None: rows = np.ceil(len(plotFunctions) / cols)
        if cols is None: cols = np.ceil(len(plotFunctions) / rows)

    assert cols*rows >= len(plotFunctions), "We r unable to fit all plots :/"        

    fig = plt.figure()
    fig.tight_layout()

    for index, plot_ in enumerate(plotFunctions): 
        if is3D is False:
            plt.subplot(int(rows), int(cols), int(index+1))
            plot_(plt)
        else:
            plot_(fig.add_subplot(int(rows), int(cols), int(index+1), projection='3d'))

    if saveFigure: 
        filename = Path("Plot_{}.png".format(title))
        path = Logger.getCurrentLogFolder() / filename
        plt.savefig(path)

    if showPlot and not MATPLOTLIB_SAVE_PLOT_ONLY: plt.show()

    return fig


def plotSurface(plt, z, rangeX, rangeY=None):
    
    XX, YY = np.meshgrid(rangeX, rangeX if rangeY is None else rangeY)
    ZZ = z(XX.flatten(), YY.flatten()).reshape(XX.shape)

    plt.plot_surface(XX, YY, ZZ, linewidth=0, antialiased=False, alpha=.3)


def getModdeTestResponse():
    # Used in Modde

    cf = np.array([.28, .14, .25, 0, .03, .13, .05, 0, .35, .35, .35, .86, .57, .63, 0.84, .41, 1, .26, .36])
    sf = np.array([.0005, .0005, 0.1988, .0005, .0356, .1394, .0227, .0002, .2771, .2773, .2813, 1.5726, .4377, .4241, .3616, .4238, .8503, .1189, .1892])
    return np.array([cf, sf]).T


def fitFactorSet(factorSet : FactorSet, Y : Iterable, verbose=True):

    X = factorSet.getExperimentValuesAndCombinations()
    scaledX = factorSet.getExperimentValuesAndCombinations(Statistics.orthogonalScaling)

    model = LR.fit(X, Y)
    sModel = LR.fit(scaledX, Y)

    if verbose: 
        scaledYPrediction = LR.predict(sModel, scaledX)
        print(sModel.summary())
        print(Statistics.getModelTermSignificance(sModel.conf_int()))
        
        Statistics.plotCoefficients(sModel.params, factorSet, sModel.conf_int())
        Statistics.plotObservedVsPredicted(scaledYPrediction, Y, X=scaledX)
        Statistics.plotResiduals(Statistics.residualsDeletedStudentized(sModel))
  
    return sModel, model

def getXWithCombinations(experimentValues : np.array, experimentValueCombinations : dict, scalingFunction : Callable = lambda x: x) -> np.array:
    scaledExperimentValues = scalingFunction(experimentValues)
    # Non scaled combinations
    combinations = np.array([
        np.array([
                func(e) for func in experimentValueCombinations.values()
            ]) for e in scaledExperimentValues
    ])

    # Combination of scaled combinations and factors
    return np.append(scaledExperimentValues, combinations, axis=1)

def removeCombinations(combinations : dict, removePredicate : Callable, factorCount):
    reduced = { key:value for index, (key,value) in  
        enumerate(combinations.items()) 
        if not removePredicate(index + factorCount + 1, key, value)
    }
    
    return reduced

def combinationIndexContainingFactor(combinations : dict, factorIndex):
    factorIndexChar = chr(65 + factorIndex % 26)
    return [combiIndex for combiIndex, combiLabel in enumerate(combinations.keys()) if factorIndexChar in combiLabel]



def getModel(experimentValues : np.array, combinations : dict, Y : np.array, scalingFunction : Callable = lambda x: x):
    X = getXWithCombinations(experimentValues, combinations, scalingFunction)
    return LR.fit(X, Y)