import point_process.poisson
import scipy.stats
import numpy as np


def thomas_point_process(kappa, sigma, mu, dx):
    """
    A Poisson( kappa ) number of parents are created,
    each forming a Poisson( mu ) numbered cluster of points,
    having an isotropic Gaussian distribution with variance `sigma`
    :param kappa: количество родителей
    :param sigma: дисперсия нормального распределения
    :param mu: пуассоновский параметр, определяющий количество выводков для каждого родителя
    :param dx: сторона квадрата
    :return:
    """
    # create a set of parent points from a Poisson( kappa )
    # distribution on the square region [0,Dx] X [0,Dx]
    parents = point_process.poisson.poisson_homogeneous_point_process(kappa, dx)

    # M is the number of parents
    m = parents.shape[0]

    # an empty list for the Thomas process points
    tp = []

    # for each parent point..
    for i in range(m):

        # determine a number of children according
        # to a Poisson( mu ) distribution
        n = scipy.stats.poisson(mu).rvs()

        # for each child point..
        for j in range(n):
            # place a point centered on the location of the parent according
            # to an isotropic Gaussian distribution with sigma variance
            sample = scipy.stats.multivariate_normal.rvs(parents[i, :2], [[sigma, 0], [0, sigma]])
            #pdf = scipy.stats.norm(loc=parents[i, :2], scale=(sigma, sigma))

            # add the child point to the list TP
            #tp.append(list(pdf.rvs(2)))
            tp.append(list(sample))

    # return a numpy array
    tp = np.array(tp)
    return tp