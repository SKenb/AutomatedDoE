from Common import Statistics
from Common import LinearRegression as LR
from Common import Common
from pathlib import Path

def generatePlot2(prediction, observation, titleStr, useLabels=True, filename=None, drawOrigin=True, drawTicks=True):

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
        saveFigure=savePath is not None,
        setFilename=filename,
        savePath=savePath
    )