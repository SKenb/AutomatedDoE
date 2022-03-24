from typing import Dict
from unittest import skip
from Common import Statistics
from Common import LinearRegression as LR
from Common import Common
from StateMachine.Context import ContextDoE
from Common.Statistics import getModelTermSignificance
from pathlib import Path
from matplotlib import colors

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
        lambda plt: plt.scatter(prediction, observation, scatterSize, zorder=200, edgecolor='k'),
        #lambda plt: plt.grid(), 
        lambda plt: plt.gca().set_aspect('equal', adjustable='box'),
        lambda plt: plt.rcParams.update({'font.size': fontSize, 'figure.autolayout': True}),
        lambda plt: drawTicks or plt.yticks([]),
        lambda plt: drawTicks or plt.xticks([]),
        lambda plt: drawTicks and plt.yticks([0, 2, 4]),
        lambda plt: drawTicks and plt.yticks([0, 2, 4]),
        #lambda plt: drawTicks and (plt.yticks([1.4, 1.6, 1.8, 2]) if filename is not None and "Rob" in filename else plt.yticks([0, .5, 1, 1.5])),
        #lambda plt: drawTicks and (plt.xticks([1.4, 1.6, 1.8, 2]) if filename is not None and "Rob" in filename else plt.xticks([0, .5, 1, 1.5])),
        xLabel=r"Predicted STY ($kg\;L^{-1}\;h^{-1}$)" if useLabels else "", 
        yLabel=r"Observed STY ($kg\;L^{-1}\;h^{-1}$)" if useLabels else "", 
        title="", 
        saveFigure=savePath is not None and figure is None,
        setFilename=filename,
        savePath=savePath,
        figure=figure,
        skip=False
    )

def plotScoreHistory(scoreHistoryDict : Dict, selectedIndex=None, drawTicks=True, useLabels=True, titleStr="", figure=False):
    def plotAllScores(p):
        color= ['blue', 'orange']

        for index, (score, scoreHistory) in enumerate(scoreHistoryDict.items()):

            p.plot(scoreHistory, label="${}$".format(score.replace("2", "^2")), lineWidth=3)
            
            if selectedIndex is not None:
                p.scatter(selectedIndex, scoreHistory[selectedIndex], 60, edgecolors='k', c=color[index], zorder=200),
 
    Common.plot(
        plotAllScores,
        lambda plt: plt.ylim((-0.1, 1.1)),
        lambda plt: plt.yticks([0, 0.5, 1]),
        lambda plt: drawTicks or plt.xticks([]),
        showLegend= len(scoreHistoryDict) > 1,
        xLabel=r"Coefficients removed" if useLabels else "", 
        yLabel=r"Score" if useLabels else "", 
        #title=("" if len(scoreHistoryDict) > 1 else scoreHistoryDict[0].keys()[0]) + "Score",
        title=titleStr,
        figure=figure,
        skip=False
    )

def plotCoefficients(coefficientValues, context:ContextDoE=None, confidenceInterval=None, titleStr = "", drawTicks=True, useLabels=True, figure=None, combinations:dict=None):
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
        labels = [r"$0$"]
        labels.extend([r"${}$".format(char(index)) for index in range(len(context.factorSet)) if not context.isFactorExcluded(index)])
        labels.extend([r"${}$".format(l.replace("*", r" \cdot ")) for l in combinations.keys()])


    def _plotBars(plt):
        bars = plt.bar(range(l), coefficientValues)
        for index, isSig in enumerate(isSignificant):
            if not isSig: bars[index].set_color('r')

    Common.plot(
        lambda plt: plt.rcParams.update({'text.usetex': True}),
        lambda plt: _plotBars(plt),
        lambda plt: plt.errorbar(range(l), coefficientValues, confidenceInterval, fmt=' ', color='b'),
        lambda plt: True if labels is None else plt.xticks(range(l), labels, rotation=90),
        #lambda plt: plt.yticks([]),
        xLabel="", 
        yLabel=r"Magnitude", 
        title=titleStr,
        figure=figure,
        skip=False
    )


def generatePlot4(prediction, context, scaledModel, combinations, combiScoreHistory, bestCombiScoreItem, useSubtitles=False, useLabels=True, filename=None, drawOrigin=True, drawTicks=True):
    savePath=Path("./Paper/Plots/Plot4_Iter/")
    sizeInCm = 10

    plt.rcParams.update({
        'text.usetex': True,
        'font.size': '18',
        'font.weight': 'bold'
    })

    Common.subplot(
        lambda fig: plotScoreHistory(
            {
                "R2": combiScoreHistory.choose(lambda i: i.r2), 
                "Q2": combiScoreHistory.choose(lambda i: i.q2)
            }, bestCombiScoreItem.index, drawTicks=drawTicks, useLabels=useLabels,
            titleStr="Score history" if useSubtitles else "", figure=fig
        ),
        lambda fig: plotCoefficients(
            scaledModel.params, context, scaledModel.conf_int(), combinations=combinations, 
            titleStr="Coefficients" if useSubtitles else "",
            drawTicks=drawTicks, useLabels=useLabels, figure=fig
        ),
        lambda fig: generatePlot2(
            prediction, context.getResponse(), titleStr="Titel1" if useSubtitles else "", 
            useLabels=useLabels, drawOrigin=False, drawTicks=True, figure=fig
        ),
        figHandler=lambda fig: fig.set_size_inches(cm2Inch(sizeInCm), cm2Inch(3*sizeInCm)),
        rows=3,
        saveFigure=True, savePath=savePath, setFilename=filename or "Plot4"
    )

def generatePlot4C(prediction, observation, titleStr, useLabels=True, filename=None, drawOrigin=True, drawTicks=True):
    return
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

def generatePlot1Bottom(r2ScoreHistory, q2ScoreHistory):
    return
    sizeInCm = 8
    savePath=Path("./Paper/Plots/Plot1_Hist/")

    fig = plt.figure()
    ax = plt.gca()

    sizeInCm = 8
    fig.set_size_inches(cm2Inch(4*sizeInCm), cm2Inch(1.5*sizeInCm))

    plt.scatter(range(len(r2ScoreHistory)), r2ScoreHistory, 60, c="tab:blue", zorder=200, label=r"$R^2$")
    plt.scatter(range(len(q2ScoreHistory)), q2ScoreHistory, 60, c="tab:orange", zorder=200, label=r"$Q^2$")

    plt.plot(r2ScoreHistory, "--", color="tab:blue")
    plt.plot(q2ScoreHistory, "--", color="tab:orange")
    
    plt.xlabel("Experiment iteration")
    plt.ylabel("Score")

    plt.legend()

    plt.savefig(savePath / Path("Plot1Bottom.png"))


def generatePlot1(experiments, factorSet, filename="ExpHist.png", useABC=True, useLabels=True, drawTicks=True):
    return
    savePath=Path("./Paper/Plots/Plot1_Hist/")

    expLabels = [f.name for f in factorSet.factors]
    if useABC: expLabels = [chr(65+(index % 26))for index in range(len(factorSet))]

    assert len(expLabels) == experiments.shape[1], "UPS 0.o - we have a different amount of labels compared to experiment variables..."

    cmap = colors.ListedColormap(['royalblue', 'lightsteelblue', 'cornflowerblue'])
    bounds=[-1.5,-.5, .5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()
    ax = plt.gca()

    sizeInCm = 8
    fig.set_size_inches(cm2Inch(4*sizeInCm), cm2Inch(sizeInCm))

    plt.imshow(experiments.T, cmap=cmap, origin='lower', norm=norm, interpolation='nearest')

    # Show all ticks and label them with the respective list entries
    if not drawTicks: plt.xticks([])
    plt.yticks(np.arange(len(expLabels)), labels=expLabels)

    if useLabels:
        #plt.xlabel("Experiments")
        plt.ylabel("Factor")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    def valueToString(val, tol=1e-3):
        if val < -tol: return "-" 
        if val > tol: return "+" 
        return "0"

    # Loop over data dimensions and create text annotations.
    for i in range(len(expLabels)):
        for j in range(experiments.shape[0]):
            text = valueToString(experiments[j, i])

            if "0" in text:
                ax.text(j, i, text, ha="center", va="center", color="w", fontsize=12)
            else:
                ax.text(j, i, text, ha="center", va="center", color="w")

    ax.xaxis.tick_top()
    ax.set_title("")
    fig.tight_layout()

    plt.savefig(savePath / Path(filename))