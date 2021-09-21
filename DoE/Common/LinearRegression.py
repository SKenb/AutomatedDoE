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
        lambda plt: plt.plot(x, y, regr.predict(np.transpose([x, y])), 'b', alpha=.3),
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

