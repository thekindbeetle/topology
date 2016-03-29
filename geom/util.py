import math

def is_obtuse( len_a, len_b, len_c ):
    """
    Является ли треугольник с заданным соотношением сторон тупоугольным
    :param len_a: длина первой стороны
    :param len_b: длина второй стороны
    :param len_c: длина третьей стороны
    :return:
    """
    if len_a >= len_b and len_a >= len_c:
        return len_b**2 + len_c**2 < len_a**2
    elif len_b >= len_c and len_b >= len_a:
        return len_c**2 + len_a**2 < len_b**2
    else:
        return len_b**2 + len_a**2 < len_c**2


def triangle_area( len_a, len_b, len_c ):
    """
    Площадь треугольника, вычисленная по сторонам по формуле Герона.
    :param len_a: длина первой стороны
    :param len_b: длина второй стороны
    :param len_c: длина третьей стороны
    :return:
    """
    p = 0.5 * float(len_a + len_b + len_c)
    return math.sqrt(p * (p - len_a) * (p - len_b) * (p - len_c))


def dist( pt_a, pt_b ):
    """
    Расстояние между двумя точками в R^2
    :param pt_a: первая точка
    :param pt_b: вторая точка
    :return:
    """
    return math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])


def inner_product( vect_1, vect_2 ):
    """
    Скалярное произведение векторов в R^2
    :param vect_1:
    :param vect_2:
    :return:
    """
    return vect_1[0] * vect_2[0] + vect_1[1] * vect_2[1]


def angle( a, b, c ):
    """
    Величина угла abc
    :param a:
    :param b:
    :param c:
    :return:
    """
    return math.acos(inner_product([a[0] - b[0], a[1] - b[1]], [c[0] - b[0], c[1] - b[1]]) / dist(b, a) / dist(b, c))


def outer_radius( pt_a, pt_b, pt_c ):
    """
    Радиус окружности, проходящей через три точки [x, y]
    :param pt_a: первая точка
    :param pt_b: вторая точка
    :param pt_c: третья точка
    :return:
    """
    return dist(pt_a, pt_b) / math.sin(angle(pt_a, pt_c, pt_b)) / 2


def outer_radius_by_sides( len_a, len_b, len_c ):
    """
    Радиус описанной окружности около треугольника с данными сторонами
    :param len_a: длина первой стороны
    :param len_b: длина второй стороны
    :param len_c: длина третьей стороны
    :return:
    """
    return float(len_a * len_b * len_c) * 0.25 / triangle_area(len_a, len_b, len_c)

def test():
    pt1 = [0, 0]
    pt2 = [2, 0]
    pt3 = [1, 2]
    print(outer_radius(pt1, pt2, pt3))