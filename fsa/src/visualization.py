# get it from http://stackoverflow.com/questions/22294241/plotting-a-decision-boundary-separating-2-classes-using-matplotlibs-pyplot

import numpy as np
from numpy import sin, cos, pi
from scipy.optimize import leastsq


def find_boundary(x, y, n, plot_pts=1000):
    def sines(theta):
        ans = np.array([sin(i * theta) for i in range(n + 1)])
        return ans

    def cosines(theta):
        ans = np.array([cos(i * theta) for i in range(n + 1)])
        return ans

    def residual(params, x, y):
        x0 = params[0]
        y0 = params[1]
        c = params[2:]

        r_pts = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5

        thetas = np.arctan2((y - y0), (x - x0))
        m = np.vstack((sines(thetas), cosines(thetas))).T
        r_bound = m.dot(c)

        delta = r_pts - r_bound
        delta[delta > 0] *= 10

        return delta

    # initial guess for x0 and y0
    x0 = x.mean()
    y0 = y.mean()

    params = np.zeros(2 + 2 * (n + 1))
    params[0] = x0
    params[1] = y0
    params[2:] += 1000

    popt, pcov = leastsq(residual, x0=params, args=(x, y),
                         ftol=1.e-12, xtol=1.e-12)

    thetas = np.linspace(0, 2 * pi, plot_pts)
    m = np.vstack((sines(thetas), cosines(thetas))).T
    c = np.array(popt[2:])
    r_bound = m.dot(c)
    x_bound = x0 + r_bound * cos(thetas)
    y_bound = y0 + r_bound * sin(thetas)

    return x_bound, y_bound
