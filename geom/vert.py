import math


class Vert:
    """
    Вершина триангуляции в R^2
    """

    # размерность
    dim = 0

    # Координаты вершины
    point = None

    # Время появления (всегда нулевое)
    appTime = 0

    # Глобальный индекс
    globInd = None

    # Индекс в фильтрации
    filtInd = None

    def __init__(self, idx, x, y):
        self.globInd = idx
        self.point = [x, y]

    def __repr__(self):
        return "Vertex #{0}, ({1}, {2})".format(self.globInd, self.point[0], self.point[1])

    def __getitem__(self, item):
        return self.point.__getitem__(item)

    def v(self, idx):
        return self.globInd

    def equals_by_global_idx(self, simplex):
        """
        Сравнение с другим симплексом по глобальному индексу.
        :param simplex: двугой симплекс
        :return:
        """
        return simplex.globInd == self.globInd

    def compare_to(self, simplex):
        """
        Сравнить с другим симплексом по времени появления в фильтрации
        :param simplex:
        :return:
        """
        return math.copysign(1, self.appTime - simplex.appTime)