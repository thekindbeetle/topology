from scipy.stats import *
import numpy as np


def poisson_homogeneous_point_process(rate, dx, dy=None, log=False, fixed_rate=False):
    """
    Реализация пуассоновского однородного точечного процесса в прямоугольной области.
    :param rate: интенсивность процесса (среднее количество точек на единицу площади)
    :param dx: длина прямоугольника
    :param dy: ширина прямоугольника (по умолчанию равна длине)
    :param log: текстовый вывод
    :param fixed_rate: Фиксировать количество создаваемых точек (будет вычислено как rate * площадь)
    :return: (x, y)
    """
    if dy is None:
        dy = dx
    point_num = int(rate * dx * dy) if fixed_rate else poisson(rate * dx * dy).rvs()
    points = uniform.rvs(0, dx, (point_num, 2))
    x = np.transpose(points)[0]
    y = np.transpose(points)[1]

    if log:
        print("{0} Poisson distributed points generated".format(point_num))

    return x, y
