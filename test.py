import matplotlib.pyplot as pyplot
import numpy as np
from scipy.stats import *
import triangle

def PoissonPP(rate, Dx, Dy=None):
    """
    Determines the number of events `N` for a rectangular region,
    given the rate `rate` and the dimensions, `Dx`, `Dy`.
    Returns a <2xN> NumPy array.
    """
    if Dy is None: Dy = Dx
    N = poisson(rate * Dx * Dy).rvs()
    print(N)
    x = uniform.rvs(0, Dx, N)
    y = uniform.rvs(0, Dy, N)
    return x, y

lamb = 100
#points = PoissonPP(lamb, 1)
points = [[1.0, 2.0, 2.0, 5.0, 9.0, 10.0, -1.0], [1.0, 1.0, 2.0, 7.0, 11.0, 11.0, 8.0]]
d = triangle.delaunay([(points[0][i], points[1][i]) for i in range(len(points[0]))])
pyplot.scatter(points[0], points[1], marker='o')
pyplot.triplot(points[0], points[1], d)
pyplot.show()