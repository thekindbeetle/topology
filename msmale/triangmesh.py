import matplotlib.pyplot as plt
import matplotlib.collections as mc
import numpy as np
import functools


class TriangMesh:
    """
    Треугольная сетка.
    Квадратная сетка, разделённая диагоналями.
    """

    def __init__(self, lx, ly, conditions='torus'):
        """
        Создание сетки с нулевыми значениями в клетках.
        :param lx: Размер сетки по X
        :param ly: Размер сетки по Y
        :return:
        """
        self.conditions = conditions  # Граничные условия
        self.sizeX = lx  # Размеры сетки по X и Y
        self.sizeY = ly
        self.size = lx * ly  # Количество вершин
        self.fields = list()  # Список полей, определённых на данной сетке
        if conditions == 'torus':
            self._generate_torus_edges()

    def _generate_torus_edges(self):
        self.hor_edges = [(row * self.sizeX + idx, row * self.sizeX + (idx + 1) % self.sizeX)
                      for idx in range(self.sizeX) for row in range(self.sizeY)]
        self.ver_edges = [(idx * self.sizeX + col, ((idx + 1) % self.sizeY) * self.sizeX + col)
                           for idx in range(self.sizeY) for col in range(self.sizeX)]
        self.diag_edges = [(idx, ((idx + 1) % self.sizeX) + ((idx // self.sizeX + 1) % self.sizeY) * self.sizeX)
                           for idx in range(self.size)]

    def set_field(self, field):
        """
        :param field: NumPy array
        """
        self.fields.append(field)

    def value(self, field_idx, vert_idx):
        """
        Значение по глобальному индексу вершины для данного поля.
        :param vert_idx: Индекс вершины.
        :param field_idx: Индекс поля.
        """
        return self.fields[field_idx][self.coordy(vert_idx), self.coordx(vert_idx)]

    def coordx(self, idx):
        """
        Координата X вершины
        """
        return idx % self.sizeX

    def coordy(self, idx):
        """
        Координата Y вершины
        """
        return idx // self.sizeX

    def edges(self):
        """
        Производящая функция рёбер.
        :return:
        """
        for edge in self.hor_edges:
            yield edge
        for edge in self.ver_edges:
            yield edge
        for edge in self.diag_edges:
            yield edge

    def _coords(self, idx):
        """
        Координаты вершины
        """
        return self.coordx(idx), self.coordy(idx)

    def _vleft(self, idx):
        """
        Левый сосед вершины с заданным индексом
        """
        return idx - idx % self.sizeX + (idx + self.sizeX - 1) % self.sizeX

    def _vright(self, idx):
        """
        Правый сосед вершины с заданным индексом
        """
        return idx - idx % self.sizeX + (idx + 1) % self.sizeX

    def _vtop(self, idx):
        """
        Верхний сосед вершины с заданным индексом
        """
        return (idx + self.size - self.sizeX) % self.size

    def _vbottom(self, idx):
        """
        Нижний сосед вершины с заданным индексом
        """
        return (idx + self.sizeX) % self.size

    def _hor_edgelink(self, edge):
        """
        Линк горизонтального ребра (две вершины для тора).
        Вызывать только для существующих рёбер.
        :param edge: Ребро.
        :return: Пара вершин — линк ребра.
        """
        return self._vtop(edge[0]), self._vbottom(edge[1])

    def _ver_edgelink(self, edge):
        """
        Линк вертикального ребра (две вершины для тора).
        Вызывать только для существующих рёбер.
        :param edge: Ребро.
        :return: Пара вершин — линк ребра.
        """
        return self._vleft(edge[0]), self._vright(edge[1])

    def _diag_edgelink(self, edge):
        """
        Линк диагонального ребра (две вершины для тора).
        Вызывать только для существующих рёбер.
        :param edge: Ребро.
        :return: Пара вершин — линк ребра.
        """
        return self._vright(edge[0]), self._vleft(edge[1])

    def cmp_jacobi_set(self, field_idx1=0, field_idx2=1, eps=0.001):
        """
        Вычисление множества Якоби для сетки на плоскости.
        Подробности алгоритма см. в статье
        Jacobi Sets of Multiple Morse Functions.
        H. Edelsbrunner, J. Harer.
        :param field_idx1: Индекс первого поля.
        :param field_idx2: Индекс второго поля.
        :return:
        """
        result = []

        def check_edge(e, link):
            if self.value(field_idx2, e[0]) - self.value(field_idx2, e[1]) < eps:
                def phi(idx):
                    return self.value(field_idx2, idx)
            else:
                l = (self.value(field_idx1, e[0]) - self.value(field_idx1, e[1]))\
                    / (self.value(field_idx2, e[0]) - self.value(field_idx2, e[1]))

                def phi(idx):
                    return self.value(field_idx1, idx) + l * self.value(field_idx2, idx)

            if (phi(e[0]) - phi(link[0])) * (phi(e[0]) - phi(link[1])) > 0:
                result.append(e)

        for edge in self.hor_edges:
            check_edge(edge, self._hor_edgelink(edge))
        for edge in self.ver_edges:
            check_edge(edge, self._ver_edgelink(edge))
        for edge in self.diag_edges:
            check_edge(edge, self._diag_edgelink(edge))

        return result

    def _is_edge_internal(self, edge):
        """
        Проверка, пересекает ли ребро границу решётки.
        :param edge: Ребро.
        :return:
        """
        return edge[0] < edge[1] and edge[0] % self.sizeX <= edge[1] % self.sizeX

    def _construct_collection(self, edges):
        """
        Построить LineCollection по набору рёбер.
        :param edges: Список рёбер.
        :return: LineCollection.
        """
        return mc.LineCollection(map(lambda e: tuple(map(self._coords, e)),
                               [e for e in edges if self._is_edge_internal(e)]), colors='k', linewidths=1)

    def draw(self, field_idx=0, draw_jacobi_set=(0, 1)):
        plt.style.use('ggplot')
        plt.imshow(self.fields[field_idx])
        #plt.gca().add_collection(self._construct_collection(list(self.edges())))
        plt.gca().add_collection(self._construct_collection(self.cmp_jacobi_set()))
        plt.show()

import msmale.field_generator as gen

field1 = np.genfromtxt(fname='../smalldata/hgt_200.csv', delimiter=' ')
field2 = np.genfromtxt(fname='../smalldata/hgt_300.csv', delimiter=' ')
# gen.perturb(field1)
# gen.perturb(field2)

t = TriangMesh(*reversed(field1.shape))
t.set_field(field1)
t.set_field(field2)
t.draw(field_idx=0)
# field2 = np.recfromcsv('smalldata/hgt_300.csv')



# t = TriangMesh(5, 5)
# t.set_field(np.zeros((5, 5)))
# t.draw()
# print(list(t.edges()))