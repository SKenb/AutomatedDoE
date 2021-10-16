from typing import Iterable
import matplotlib.pyplot as plt
import Common.LinearRegression as LR
import numpy as np
import Common.Statistics as Statistics
import logging

from datetime import datetime
from pathlib import Path

from Common.Factor import FactorSet


def plot(*plotters, is3D=False, xLabel="x", yLabel="y", title="Plot", showLegend=False):
    
    figure = plt.figure()

    if is3D: ax = figure.add_subplot(111, projection='3d')
    for plotter in plotters: plotter(ax if is3D else plt)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)

    if showLegend: plt.legend()
    
    plt.show()

    return figure


def plotSurface(plt, z, rangeX, rangeY=None):
    
    XX, YY = np.meshgrid(rangeX, rangeX if rangeY is None else rangeY)
    ZZ = z(XX.flatten(), YY.flatten()).reshape(XX.shape)

    plt.plot_surface(XX, YY, ZZ, linewidth=0, antialiased=False, alpha=.3)

# Initialize logging in one defines place
# import logging in all other files
#
# Use:
#   - logging.debug('...')
#   - logging.info('...')
#   - logging.warning('...')
#   - logging.error('...')
#
def initLogging():

    logPath = Path("./Logs/log_{}.log".format(datetime.now().strftime("%d%m%Y_%H")))

    logging.basicConfig(
        filename=str(logPath), 
        format='%(asctime)s\t%(levelname)s\t%(message)s',
        datefmt='%d.%m.%Y %I:%M:%S %p',
        level=logging.DEBUG
    )

def getModdeTestResponse():
    # Used in Modde

    cf = np.array([.28, .14, .25, 0, .03, .13, .05, 0, .35, .35, .35, .86, .57, .63, 0.84, .41, 1, .26, .36])
    sf = np.array([.0005, .0005, 0.1988, .0005, .0356, .1394, .0227, .0002, .2771, .2773, .2813, 1.5726, .4377, .4241, .3616, .4238, .8503, .1189, .1892])
    return np.array([cf, sf]).T


def fitFactorSet(factorSet : FactorSet, Y : Iterable, verbose=True):

    X = factorSet.getExperimentValuesAndCombinations()
    scaledX = factorSet.getExperimentValuesAndCombinations(Statistics.orthogonalScaling)
    print(scaledX)


    model = LR.fit(X, Y)
    sModel = LR.fit(scaledX, Y)

    if verbose: 
        scaledYPrediction = LR.predict(sModel, scaledX)
        print(sModel.summary())
        print(Statistics.Q2(scaledX, Y, scaledYPrediction))
        print(Statistics.getModelTermSignificance(sModel.conf_int()))
        
        Statistics.plotCoefficients(sModel.params, factorSet, sModel.conf_int())
        Statistics.plotObservedVsPredicted(scaledYPrediction, Y, X=scaledX)
        Statistics.plotResiduals(Statistics.residualsDeletedStudentized(sModel))
  
    return sModel, model