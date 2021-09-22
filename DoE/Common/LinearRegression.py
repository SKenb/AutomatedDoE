from scipy.optimize.optimize import main
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

import Common

def LinearRegression(X, y, kernel=None):
    if kernel is None: return linear_model.LinearRegression().fit(X, y)
    return make_pipeline(kernel, linear_model.LinearRegression()).fit(x, y)


if __name__ == '__main__':
    
    print(" Test Linear Regression wrapper functions ".center(80, "-"))
    
    # Linear Regression for 1 dim
    x = np.arange(-1, 5, .25)
    y = 3*(x-2) + 5*(np.random.rand(len(x)) -.5)

    regr = LinearRegression(x.reshape(-1, 1), y)

    Common.plot(
        lambda plt: plt.plot(x, y, 'r.'),
        lambda plt: plt.plot(x, regr.predict(x.reshape(-1, 1)), 'b-'), 
    )

    # Linear Regression for 2 dim
    x = np.random.random(100)*10+20
    y = np.random.random(100)*5+7
    z = 5*x+3*y + 20*(np.random.rand(len(x)) -.5)

    regr = LinearRegression(np.transpose([x, y]), z)

    Common.plot(
        lambda plt: plt.plot(x, y, z, 'r.'),
        lambda plt: Common.plotSurface(
            plt,
            lambda x, y: regr.predict(np.transpose([x, y])), 
            np.arange(20, 30, .25), np.arange(7, 12, .25)
        ),
        is3D=True
    )


    # Linear Regression using kernel stuff
    x = np.arange(-1, 5, .25)[:, None]
    y = np.ravel(x) ** 2 + 10*(np.random.rand(len(x)) -.5)

    regr = LinearRegression(x, y, PolynomialFeatures(2))

    Common.plot(
        lambda plt: plt.plot(x, y, 'r.'),
        lambda plt: plt.plot(x, regr.predict(x.reshape(-1, 1)), 'b-'), 
    )

    # Linear Regression 2dim using man. kernel stuff
    x = np.random.random(300)*20-10
    y = np.random.random(300)*20-10
    z = .5*x**2 + 2*y**2 + 10*x + 4*y + x*y + 100*(np.random.rand(len(x)) -.5)

    regr = LinearRegression(np.transpose([x, x**2, y, y**2, x*y]), z)
    
    Common.plot(
        lambda plt: plt.plot(x, y, z,  'r.'),
        lambda plt: Common.plotSurface(plt, 
            lambda x, y: regr.predict(np.transpose([x, x**2, y, y**2, x*y])), 
            np.arange(-10, 10, .25)
        ),
        is3D=True 
    )

