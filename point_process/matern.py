import scipy.stats
import numpy as np
import point_process.poisson
import matplotlib.pyplot as plt


def uniform_disk(x, y, r):
    """
    Равномерно распределённая в круге точка
    :param x: координата X центра круга
    :param y: координата Y центра круга
    :param r: радиус круга
    :return:
    """
    r = scipy.stats.uniform(0, r ** 2.0).rvs()
    theta = scipy.stats.uniform(0, 2 * np.pi).rvs()
    xt = np.sqrt(r) * np.cos(theta)
    yt = np.sqrt(r) * np.sin(theta)
    return x + xt, y + yt


def matern_point_process(kappa, r, mu, dx, include_parents=False, logging_on=True):
    """
    A Poisson( kappa ) number of parents are created,
    each forming a Poisson( mu ) numbered cluster of points,
    distributed uniformly in a circle of radius `r`
    :param kappa: параметр Пуассоновского распределения, определяющего количество точек в единичном квадрате
    :param r: радиус круга, в котором для каждого родителя создаются потомки
    :param mu: параметр Пуассоновского распределения, определяющего количество потомков
    :param dx: длина стороны квадрата
    :param include_parents: вывести родительские события
    :param logging_on: текстовый вывод
    :return:
    """
    # create a set of parent points from a Poisson( kappa )
    # distribution on the square region [0,Dx] X [0,Dx]
    parents = point_process.poisson.poisson_homogeneous_point_process(kappa, dx)

    # M is the number of parents
    m = parents.shape[0]

    # an empty list for the Matern process points
    matern_points = list()

    # for each parent point..
    for i in range(m):

        # determine a number of children according
        # to a Poisson( mu ) distribution
        n = scipy.stats.poisson(mu).rvs()

        # for each child point..
        for j in range(n):

            # produce a uniformly distributed child point from a
            # circle of radius `r`, centered about a parent point
            x, y = uniform_disk(parents[i, 0], parents[i, 1], r)

            # add the child point to the list MP
            matern_points.append([x, y])

    # return a numpy array
    matern_points = np.array(matern_points)
    if include_parents:
        if logging_on:
            print("{0} Matern distributed points generated.".format(len(matern_points) + len(parents)))
        return matern_points, parents
    else:
        if logging_on:
            print("{0} Matern distributed points generated.".format(len(matern_points)))
        return matern_points
