import matplotlib.pyplot as plt
import numpy as np
import logging

from datetime import datetime
from pathlib import Path


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