import matplotlib.pyplot as plt

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