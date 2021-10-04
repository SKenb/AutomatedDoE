from Common import Common

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
        lambda plt: plt.text(.5*(minVal + maxVal), 0.1, "R2: {}".format(round(r2_score(observation, prediction), 5))),
        xLabel="Predicted", yLabel="Observed", title=titleStr
    )
