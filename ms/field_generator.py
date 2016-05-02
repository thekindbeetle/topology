import numpy as np
import numpy.random
import scipy.stats


def gen_gaussian_sum(str, col, centers, sigma):
    """
    Матрица, являющаяся суммой гауссиан
    :param str: количество строк матрицы
    :param col: количество столбцов матрицы
    :param centers: список центров (не обязательно целочисленных, могут даже лежать за пределами матрицы!)
    :param sigma: ширина гауссиан
    """
    field = np.zeros((str, col))
    for center in centers:
        for i in range(str):
            for j in range(col):
                field[i, j] += scipy.stats.multivariate_normal.pdf((i, j), mean=center, cov=((sigma, 0), (0, sigma)))
    print("Field {0}x{1} generated as sum of {2} gaussians".format(str, col, len(centers)))
    return field


def gen_sincos_field(str, col, kX, kY):
    """
    Матрица функции sin (x * kX) + cos (y * kY)
    :param str: количество строк матрицы
    :param col: количество столбцов матрицы
    :param kX: коэффициент по X
    :param kY: коэффициент по Y
    :return:
    """
    func = lambda x, y: np.sin(x * kX) + np.cos(y * kY)
    field = np.zeros((str, col))
    for i in range(str):
        for j in range(col):
            field[i, j] = func(i, j)
    return field


def perturb(field, eps=0.000001):
    """
    Возмущение матрицы. На выходе - матрица, сохраняющая все топологические особенности, все значения которой различны.
    :param field: Матрица (numpy.ndarray)
    :param eps: Значения, отличающиеся меньше чем на eps, считаем равными
    :return:
    """
    flat = sorted(field.flatten())
    dif = min([np.abs(flat[i + 1] - flat[i]) for i in range(len(flat) - 1) if np.abs(flat[i + 1] - flat[i]) > eps])
    print("Minimal difference is {0}".format(dif))
    lx = field.shape[0]
    ly = field.shape[1]
    for i in range(lx):
        for j in range(ly):
            field[i, j] += dif * (i + lx * j) / (2 * lx * ly)


def perturb2(field):
    lx = field.shape[0]
    ly = field.shape[1]
    field += numpy.random.rand(lx, ly)