from scipy.stats import *
import numpy as np


def poisson_homogeneous_point_process(rate, dx, dy=None):
    """
    Реализация пуассоновского однородного точечного процесса в прямоугольной области.
    :param rate: интенсивность процесса (среднее количество точек на единицу площади)
    :param dx: длина прямоугольника
    :param dy: ширина прямоугольника (по умолчанию равна длине)
    :return: список точек из R^2
    """
    if dy is None:
        dy = dx
    point_num = poisson(rate * dx * dy).rvs()
    x = uniform.rvs(0, dx, ((point_num, 1)))
    y = uniform.rvs(0, dy, ((point_num, 1)))
    out = np.hstack((x, y))
    return out
