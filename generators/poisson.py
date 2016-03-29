from scipy.stats import *


def poisson_homogeneous_point_process(rate, dx, dy=None):
    """
    Реализация пуассоновского однородного точечного процесса в прямоугольной области.
    :param rate: интенсивность процесса (среднее количество точек на единицу площади)
    :param dx: длина прямоугольника
    :param dy: ширина прямоугольника (по умолчанию равна длине)
    :return: список точек из R^2
    """
    if dy is None: dy = dx
    point_num = poisson(rate * dx * dy).rvs()
    x = uniform.rvs(0, dx, point_num)
    y = uniform.rvs(0, dy, point_num)
    out = [[x[i], y[i]] for i in range(point_num)]
    print("{0} Poisson points generated.".format(point_num))
    return out
