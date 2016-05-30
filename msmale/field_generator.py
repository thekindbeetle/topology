import numpy as np
import numpy.random
import scipy.stats
from PIL import Image


def gen_gaussian_sum_rectangle(str, col, centers, sigma):
    """
    Матрица, являющаяся суммой гауссиан (на прямоугольнике)
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


def gen_gaussian_sum_torus(str, col, centers, sigma):
    """
    Матрица, являющаяся суммой гауссиан на торе.
    Считаем значения гауссиан за пределами 3-сигма нулевыми.
    :param str: количество строк матрицы
    :param col: количество столбцов матрицы
    :param centers: список центров (не обязательно целочисленных, могут даже лежать за пределами матрицы!)
    :param sigma: ширина гауссиан
    :return:
    """
    field = np.zeros((str, col))
    sigma3 = int(sigma * 3)
    gaussian = np.zeros((sigma3 * 2 + 1, sigma3 * 2 + 1))
    for i in range(-sigma3, sigma3 + 1):
        for j in range(-sigma3, sigma3 + 1):
            if i ** 2 + j ** 2 <= sigma3:
                gaussian[i, j] += scipy.stats.multivariate_normal.pdf((i, j), mean=(0, 0), cov=((sigma, 0), (0, sigma)))
    for center in centers:
        for i in range(-sigma3, sigma3 + 1):
            for j in range(-sigma3, sigma3 + 1):
                field[(int(center[0]) + i) % str, (int(center[1]) + j) % str] += gaussian[i, j]
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


def gen_bmp_field(fname):
    """
    Построить сетку по BMP-изображению.
    В узлах значения яркости в пикселе.
    :param fname:
    :return:
    """
    with Image.open(fname) as image:
        image = image.convert('L')
        return np.array(image, dtype=float)


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