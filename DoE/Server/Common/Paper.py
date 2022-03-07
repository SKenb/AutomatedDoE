from typing import Dict
from Common import Statistics
from Common import LinearRegression as LR
from Common import Common
from StateMachine.Context import ContextDoE
from Common.Statistics import getModelTermSignificance
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def cm2Inch(cm): 
    return cm/2.54

def generatePlot2(prediction, observation, titleStr, useLabels=True, filename=None, drawOrigin=True, drawTicks=True, figure=None):

    savePath=Path("./Paper/Plots/Plot2_ObsVsPred/")

    red = lambda func: func(func(prediction), func(observation))

    minVal = red(min)
    maxVal = red(max)

    lineWidth = 3
    fontSize = 16
    scatterSize = 160

    Common.plot(
        lambda plt: drawOrigin and plt.plot([minVal, maxVal], [minVal, minVal], 'k', linewidth=lineWidth) ,
        lambda plt: drawOrigin and plt.plot([minVal, minVal], [minVal, maxVal], 'k', linewidth=lineWidth),
        lambda plt: plt.plot([minVal, maxVal], [minVal, maxVal], 'k--', linewidth=lineWidth),
        lambda plt: plt.scatter(prediction, observation, scatterSize, zorder=200),
        lambda plt: plt.grid(), 
        lambda plt: plt.axis('equal'),
        lambda plt: plt.rcParams.update({'font.size': fontSize}),
        lambda plt: drawTicks or plt.yticks([]),
        lambda plt: drawTicks or plt.xticks([]),
        xLabel="Predicted" if useLabels else "", 
        yLabel="Observed" if useLabels else "", 
        title=titleStr, 
        saveFigure=savePath is not None and figure is None,
        setFilename=filename,
        savePath=savePath,
        figure=figure
    )

def plotScoreHistory(scoreHistoryDict : Dict, selectedIndex=None, drawTicks=True, useLabels=True, titleStr="R2 and Q2", figure=False):

    def plotAllScores(p):
        for _, (score, scoreHistory) in enumerate(scoreHistoryDict.items()):
            p.plot(scoreHistory, label=score, lineWidth=3)
            
            if selectedIndex is not None:
                p.scatter(selectedIndex, scoreHistory[selectedIndex], 60, color='r', zorder=200),
 
    Common.plot(
        plotAllScores,
        lambda plt: drawTicks or plt.yticks([]),
        lambda plt: drawTicks or plt.xticks([]),
        showLegend= len(scoreHistoryDict) > 1,
        xLabel="Iteration" if useLabels else "", 
        yLabel="Score" if useLabels else "", 
        #title=("" if len(scoreHistoryDict) > 1 else scoreHistoryDict[0].keys()[0]) + "Score",
        title=titleStr,
        figure=figure
    )

def plotCoefficients(coefficientValues, context:ContextDoE=None, confidenceInterval=None, titleStr = "Coefficients plot", drawTicks=True, useLabels=True, figure=None, combinations:dict=None):
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
        labels.extend(combinations.keys())


    def _plotBars(plt):
        bars = plt.bar(range(l), coefficientValues)
        for index, isSig in enumerate(isSignificant):
            if not isSig: bars[index].set_color('r')

    Common.plot(
        lambda plt: _plotBars(plt),
        lambda plt: plt.errorbar(range(l), coefficientValues, confidenceInterval, fmt=' ', color='b'),
        lambda plt: True if labels is None else plt.xticks(range(l), labels, rotation=90),
        lambda plt: drawTicks or plt.yticks([]),
        lambda plt: drawTicks or plt.xticks([]),
        xLabel="Coefficient" if useLabels else "", 
        yLabel="Value" if useLabels else "", 
        title=titleStr,
        figure=figure
    )


def generatePlot4(prediction, context, scaledModel, combinations, combiScoreHistory, bestCombiScoreItem, useSubtitles=False, useLabels=True, filename=None, drawOrigin=True, drawTicks=True):
    savePath=Path("./Paper/Plots/Plot4_Iter/")
    sizeInCm = 10

    Common.subplot(
        lambda fig: generatePlot2(
            prediction, context.getResponse(), titleStr="Titel1" if useSubtitles else "", 
            useLabels=useLabels, drawOrigin=drawOrigin, drawTicks=drawTicks, figure=fig
        ),
        lambda fig: plotCoefficients(
            scaledModel.params, context, scaledModel.conf_int(), combinations=combinations, 
            titleStr="Coefficients" if useSubtitles else "",
            drawTicks=drawTicks, useLabels=useLabels, figure=fig
        ),
        lambda fig: plotScoreHistory(
            {
                "R2": combiScoreHistory.choose(lambda i: i.r2), 
                "Q2": combiScoreHistory.choose(lambda i: i.q2)
            }, bestCombiScoreItem.index, drawTicks=drawTicks, useLabels=useLabels,
            titleStr="Score history" if useSubtitles else "", figure=fig
        ),
        figHandler=lambda fig: fig.set_size_inches(cm2Inch(sizeInCm), cm2Inch(3*sizeInCm)),
        rows=3,
        saveFigure=True, savePath=savePath, setFilename=filename or "Plot4"
    )

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