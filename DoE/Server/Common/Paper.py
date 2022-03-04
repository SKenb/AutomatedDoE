from Common import Statistics
from Common import LinearRegression as LR
from Common import Common
from pathlib import Path

import matplotlib.pyplot as plt

def generatePlot2(prediction, observation, titleStr, useLabels=True, filename=None, drawOrigin=True, drawTicks=True, figure=None):

    savePath=Path("./Paper/Plots/Plot2_ObsVsPred/")

    red = lambda func: func(func(prediction), func(observation))

    minVal = red(min)
    maxVal = red(max)

    lineWidth = 3
    fontSize = 16
    scatterSize = 60

    Common.plot(
        lambda plt: drawOrigin and plt.plot([minVal, maxVal], [minVal, minVal], 'k', linewidth=lineWidth) ,
        lambda plt: drawOrigin and plt.plot([minVal, minVal], [minVal, maxVal], 'k', linewidth=lineWidth),
        lambda plt: plt.plot([minVal, maxVal], [minVal, maxVal], 'k--', linewidth=lineWidth),
        lambda plt: plt.scatter(prediction, observation, scatterSize, zorder=200),
        lambda plt: plt.grid(), 
        lambda plt: plt.axis('equal'),
        lambda plt: plt.rcParams.update({'font.size': fontSize}),
        lambda plt: drawTicks and plt.yticks([]),
        lambda plt: drawTicks and plt.xticks([]),
        xLabel="Predicted" if useLabels else "", 
        yLabel="Observed" if useLabels else "", 
        title=titleStr, 
        saveFigure=savePath is not None and figure is None,
        setFilename=filename,
        savePath=savePath,
        figure=figure
    )

def generatePlot4(prediction, observation, titleStr, useLabels=True, filename=None, drawOrigin=True, drawTicks=True):
    savePath=Path("./Paper/Plots/Plot4_Iter/")

def generatePlot4C(prediction, observation, titleStr, useLabels=True, filename=None, drawOrigin=True, drawTicks=True):
    
    savePath=Path("./Paper/Plots/Plot4_Iter/")
    
    N = 3
    pattern = []
    for largeSection in ['ObsVsPred', 'Coeff', 'Scores']:
        for index in range(N):
            pattern.append(N*[largeSection])

    for index in range(N):
        pattern.append(['Contour_' + str(index*N+inner) for inner in range(N)])

    fig, axd = plt.subplot_mosaic(pattern, constrained_layout=True)

    generatePlot2(prediction, observation, "OvP", useLabels, drawOrigin, drawTicks, figure=axd['ObsVsPred'])
    generatePlot2(prediction, observation, "C", useLabels, drawOrigin, True, figure=axd['Coeff'])
    generatePlot2(prediction, observation, "Sc", True, True, drawTicks, figure=axd['Scores'])

    #for index in range(N*N):
    #    generatePlot2(prediction, observation, titleStr, useLabels, drawOrigin, drawTicks, figure=axd['Contour_'+str(index)])

    plt.savefig(savePath / Path("Iter") if filename is None else Path(filename))
    plt.show()

    exit()