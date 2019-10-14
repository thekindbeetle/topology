import math
import triangulation.util


def test():
    a1 = 8
    b1 = 7
    a2 = 3
    b2 = 6
    e1 = Edge( 1, a1, b1 )
    e2 = Edge( 2, a2, b2 )
    print(e1)
    print(e2)
    print(e1.equals_by_global_idx(e1))
    print(e1.equals_by_global_idx(e2))
    print(e1.compare_to(e1))
    print(e1.compare_to(e2))


class Edge:
    v0 = None
    v1 = None
    appTime = None
    globInd = None
    filtInd = None
    dim = 1

    @staticmethod
    def contains_edge( edge_list, v0, v1 ):
        """
        Проверка, содержится ли ребро [v0, v1] в списке edge_list
        :param edge_list:
        :param v0: первая вершина ребра
        :param v1: вторая вершина ребра
        :return:
        """
        for e in edge_list:
            if e.equals(v0, v1):
                return True
        return False

    def __init__(self, idx, v0=None, v1=None):
        self.globInd = idx
        self.v0 = v0
        self.v1 = v1

    def __repr__(self):
        return "Edge #{0}, [{1}, {2}]".format(self.globInd, self.v0, self.v1)

    def equals(self, v0, v1):
        """
        Проверка, являются ли v0 и v1 вершинами данного ребра
        :param v0:
        :param v1:
        :return:
        """
        return (self.v0 == v0 and self.v1 == v1) or (self.v0 == v1 and self.v1 == v0)

    def v( self, idx ):
        return self.v0 if idx == 0 else self.v1

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


test()
