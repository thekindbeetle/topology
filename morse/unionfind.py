import numpy as np


class UnionFind:
    """
    Union-Find data structure.
    """

    _size = 0

    # Массив родителей элементов в лесу.
    # Если родитель равен элементу, то элемент — корень дерева.
    _parent = []

    # Глубина узла в дереве.
    # Используется для балансировки дерева.
    _rank = []

    def __init__(self, size):
        self._size = size
        self._parent = [0] * size
        self._rank = [0] * size

    def makeset(self, x):
        """
        Создать новый класс с представителем x.
        :param x: представитель нового класса
        :return:
        """
        self._parent[x] = x
        self._rank[x] = 0

    def union(self, x, y):
        """
        Объединение двух классов с представителями x и y,
        x назначается представителем нового класса.
        :param x: представитель класса, в который вливается другой класс
        :param y: представитель вливаемого класса
        :return:
        """
        root_x = self.find(x)
        root_y = self.find(y)

        # Eсли элементы принадлежат к одному классу, то всё хорошо
        if root_x != root_y:
            # Прикрепляем дерево с меньшей глубиной к дереву с большей глубиной
            if self._rank[root_x] < self._rank[root_y]:
                self._parent[root_x] = root_y
            else:
                self._parent[root_y] = root_x

            # если глубина дерева увеличилась
            if self._rank[root_x] == self._rank[root_y]:
                self._rank[root_x] += 1

    def find(self, x):
        """
        Определить класс, к которому принадлежит элемент x.
        Рекурсивно поднимаемся по дереву до корня.
        :param x:
        :return:
        """
        return x if x == self._parent[x] else self.find(self._parent[x])

    def __repr__(self):
        return str(self._parent)

def test():
    uf = UnionFind(10)
    for i in range(10):
        uf.makeset(i)
    print(uf)
    uf.union(2, 4)
    uf.union(2, 7)
    uf.union(7, 4)
    uf.union(9, 5)
    uf.union(1, 5)
    print(uf.find(5))
    print(uf.find(7))
    uf.union(5, 7)
    print(uf.find(5))
    print(uf)