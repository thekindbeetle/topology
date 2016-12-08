import math
import geom.util
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import geom.vert
import geom.edge
import geom.triang
import triangle
from operator import attrgetter


class Filtration:
    """
    Фильтрация Чеха для данного множества вершин в R^2
    """

#   ---------- Вершины -----------  #

    # список вершин (Vert)
    vertices = None

    # список списков инцидентных рёбер
    # вершине с индексом i соответствует список [i] инцидентных рёбер
    incidEdgesToVertices = None

    # список списков инцидентных треугольников
    # вершине с индексом i соответствует список [i] инцидентных треугольников
    incidTrianglesToVertices = None

    # список граничных вершин триангуляции
    boardVertices = None


#  --------------- Рёбра ---------------  #

    # список рёбер триангуляции
    edges = None

    # список списков инцидентных треугольников
    # ребру с индексом i соответствует список [i] инцидентных треугольников
    incidTrianglesToEdges = None

    # список граничных рёбер триангуляции
    boardEdges = None

#  --------------- Треугольники ---------------  #

    # список треугольников триангуляции
    triangles = None

    # список списков инцидентных рёбер
    incidEdgesToTriangles = None



    # Список симплексов фильтрации
    simplexes = None

    # Количество вершин
    vertNum = None

    # Количество рёбер
    edgeNum = None

    # Количество треугольников
    trNum = None

    def __init__(self, points):
        """
        Построение фильтрации Чеха по набору точек из R^2.
        :param points:
        :return:
        """
        self.vertNum = len(points)

        # Индексирование вершин, рёбер и треугольников
        self.vertices = [geom.vert.Vert(idx, points[idx][0], points[idx][1]) for idx in range(self.vertNum)]
        self.incidEdgesToVertices = [[] for i in range(self.vertNum)]
        self.incidTrianglesToVertices = [[] for i in range(self.vertNum)]

        tr = triangle.delaunay(points)
        self.trNum = len(tr)

        self.triangles = [geom.triang.Triang(idx, tr[idx][0], tr[idx][1], tr[idx][2]) for idx in range(self.trNum)]
        self.incidEdgesToTriangles = [[] for i in range(self.trNum)]

        # Инициализируем список рёбер
        self.edges = []

        # Индексируем рёбра для быстрого поиска
        edges_idx = dict()

        idx = 0

        # внешность не учитываем
        for triIdx in range(self.trNum):
            tr = self.triangles[triIdx]
            if not edges_idx.get((tr.v(0), tr.v(1))):
                self.edges.append(geom.edge.Edge(idx, tr.v(0), tr.v(1)))
                edges_idx[(tr.v(0), tr.v(1))] = True
                edges_idx[(tr.v(1), tr.v(0))] = True
                idx += 1
            if not edges_idx.get((tr.v(0), tr.v(2))):
                self.edges.append(geom.edge.Edge(idx, tr.v(0), tr.v(2)))
                edges_idx[(tr.v(0), tr.v(2))] = True
                edges_idx[(tr.v(2), tr.v(0))] = True
                idx += 1
            if not edges_idx.get((tr.v(1), tr.v(2))):
                self.edges.append(geom.edge.Edge(idx, tr.v(1), tr.v(2)))
                edges_idx[(tr.v(1), tr.v(2))] = True
                edges_idx[(tr.v(2), tr.v(1))] = True
                idx += 1

        self.edgeNum = len(self.edges)
        self.incidTrianglesToEdges = [[] for i in range(self.edgeNum)]

        # Инициализация списков инцидентных рёбер для вершин
        # Пробегаем все ребра, каждое приписываем двум его вершинам.
        # Результат: массив, индексы которого - глоальные номера точек,
        # в i-ой ячейке - список индексов ребер, инцедентных i-ой вершине.
        for idx in range(self.edgeNum):
            v0 = self.edges[idx].v(0)
            v1 = self.edges[idx].v(1)
            self.incidEdgesToVertices[v0].append(idx)
            self.incidEdgesToVertices[v1].append(idx)

        # Инициализация списков инцидентных треугольников для вершин
        # Пробегаем все треугольники, каждый записываем в 3 списка,
        # соответствующих вершинам этого треугольника.
        # Результат: массив, индексы которого - глоальные номера точек,
        # в i-й ячейке - список индексов треугольников, содержащих i-ую вершину.
        for idx in range(self.trNum):
            self.incidTrianglesToVertices[self.triangles[idx].v(0)].append(idx)
            self.incidTrianglesToVertices[self.triangles[idx].v(1)].append(idx)
            self.incidTrianglesToVertices[self.triangles[idx].v(2)].append(idx)

        # Инициализация списков инцидентных рёбер для треугольников
        # Пробегаем все треугольники
        for i in range(self.trNum):
            # Количество ребер, инцидентных вершине A треугольника i.
            iAEdgesCount = len(self.incidEdgesToVertices[self.triangles[i].v(0)])
            # Для i-ого треугольника просматриваем список инцедентных ребер вершины A.
            # Если текущее (j-ое) ребро равно ребру AB или AC i-ого треугольника,
            # добавляем это ребро в список ребер, инцедентных i-ому треугольнику.
            for j in range(iAEdgesCount):
                # Глобальный индекс j-ого ребра инцидентного вершине A i-ого треугольника.
                iAj = self.incidEdgesToVertices[self.triangles[i].v(0)][j]
                if(self.edges[iAj].equals(self.triangles[i].v(0), self.triangles[i].v(1)) or
                   self.edges[iAj].equals(self.triangles[i].v(0), self.triangles[i].v(2))):
                    self.incidEdgesToTriangles[i].append(iAj)
            # Количество ребер, инцидентных вершине B треугольника i.
            iBEdgesCount = len(self.incidEdgesToVertices[self.triangles[i].v(1)])
            # Для i-ого треугольника просматриваем список инцедентных ребер вершины B.
            # Если текущее (j-ое) ребро равно ребру BC i-ого треугольника,
            # добавляем это ребро в список ребер, инцедентных i-ому треугольнику.
            for j in range(iBEdgesCount):
                # Глобальный индекс j-ого ребра инцидентного вершине B i-ого треугольника.
                iBj = self.incidEdgesToVertices[self.triangles[i].v(1)][j]
                if self.edges[iBj].equals(self.triangles[i].v(1), self.triangles[i].v(2)):
                    self.incidEdgesToTriangles[i].append(iBj)

        # Инициализация списка инцидентных треугольников для рёбер
        # Пробегаем все треугольники, каждый записываем в 3 списка,
        # соответствующих ребрам этого треугольника.
        # Результат: массив, индексы которого - глобальные номера ребер,
        # в i-ой ячейке - список индексов треугольников, инцедентных i-ому ребру.
        for i in range(self.trNum):
            self.incidTrianglesToEdges[self.incidEdgesToTriangles[i][0]].append(i)
            self.incidTrianglesToEdges[self.incidEdgesToTriangles[i][1]].append(i)
            self.incidTrianglesToEdges[self.incidEdgesToTriangles[i][2]].append(i)

        # Инициализация списка граничных рёбер
        self.boardEdges = []
        for edge_idx in range(self.edgeNum):
            if len(self.incidTrianglesToEdges[edge_idx]) == 1:
                self.boardEdges.append(edge_idx)

        # Инициализация списка граничных вершин
        self.boardVertices = []
        for edge_idx in self.boardEdges:
            v0 = self.edges[edge_idx].v(0)
            v1 = self.edges[edge_idx].v(1)
            if v0 not in self.boardVertices:
                self.boardVertices.append(v0)
            if v1 not in self.boardVertices:
                self.boardVertices.append(v1)

        # Add outer face to the triangulation
        outIdx = self.trNum
        out = geom.triang.Out(outIdx, self.boardVertices)
        self.triangles.append(out)
        self.incidEdgesToTriangles.append(self.boardEdges)

        for vert_idx in self.boardVertices:
            self.incidTrianglesToVertices[vert_idx].append(outIdx)

        for edge_idx in self.boardEdges:
            self.incidTrianglesToEdges[edge_idx].append(outIdx)

        self.trNum += 1  # учитываем внешность

        self.simpNum = self.vertNum + self.edgeNum + self.trNum

        self.simplexes = []

        # Добавление вершин, ребер, треугольников, внешности
        for i in range(self.vertNum):
            self.simplexes.append(self.vertices[i])
        for i in range(self.edgeNum):
            self.simplexes.append(self.edges[i])
        for i in range(self.trNum):
            self.simplexes.append(self.triangles[i])

        # Инициализация времен появления
        for s in self.simplexes:
            if s.dim == 0:
                s.appTime = 0
            elif s.dim == 1:
                length = geom.util.dist(self.vertices[s.v(0)], self.vertices[s.v(1)])
                s.appTime = length / 2
            elif s.dim == 2:
                len_a = geom.util.dist(self.vertices[s.v(0)], self.vertices[s.v(1)])
                len_b = geom.util.dist(self.vertices[s.v(0)], self.vertices[s.v(2)])
                len_c = geom.util.dist(self.vertices[s.v(1)], self.vertices[s.v(2)])
                if geom.util.is_obtuse(len_a, len_b, len_c):
                    s.appTime = max([len_a, len_b, len_c]) / 2
                else:
                    s.appTime = geom.util.outer_radius(self.vertices[s.v(0)],
                                                       self.vertices[s.v(1)],
                                                       self.vertices[s.v(2)])
        self.simplexes[-1].appTime = max([self.triangles[i].appTime for i in range(self.trNum - 1)]) + 1

        # Сортировка списка симплексов по времени появления
        self.sort_simplexes()

        # Инициализация индексов фильтрации симплексов
        for i in range(self.simpNum):
            self.simplexes[i].filtInd = i

    def get_simplex(self, filtr_index):
        """
        Get simplex by filtration index
        :param filtr_index: filtration index
        :return:
        """
        return self.simplexes[filtr_index]

    def get_min_app_time(self):
        """
        Время появления первого ребра фильтрации
        :return:
        """
        return self.simplexes[self.vertNum].appTime

    def get_max_app_time(self):
        """
        Время появления последнего симплекса фильтрации (за исключением внешности)
        :return:
        """
        return self.simplexes[len(self.simplexes) - 2].appTime

    def simplexes_num(self):
        return len(self.simplexes)

    def get_inc_triang_of_edge(self, edge_filt_idx):
        edge = self.get_simplex(edge_filt_idx)
        tr_glob_indexes = self.incidTrianglesToEdges[edge.globInd]
        global_tr_idx_0 = tr_glob_indexes[0]
        global_tr_idx_1 = tr_glob_indexes[1]
        filt_tr_idx_0 = self.triangles[global_tr_idx_0].filtInd
        filt_tr_idx_1 = self.triangles[global_tr_idx_1].filtInd
        return [filt_tr_idx_0, filt_tr_idx_1]

    def sort_simplexes(self):
        """
        Сортировка симплексов по времени появления.
        Важно! Используется устойчивая сортировка.
        Поскольку в исходном списке треугольники идут после рёбер,
        треугольники будут идти после рёбер с одинаковым временем появления.
        :return:
        """
        self.simplexes.sort(key=attrgetter('appTime'))

    def print(self):
        print("Filtration")
        for s in self.simplexes:
            print("f.ind: {0}, appearance time = {1}, {2}".format(s.filtInd, s.appTime, s))

    def print_min_max(self):
        print("Minimal appearance time: {0}".format(self.get_min_app_time()))
        print("Maximal appearance time: {0}".format(self.get_max_app_time()))

    def draw(self):
        """
        Отображение фильтрации в виде триангуляции.
        :return:
        """
        plt.figure()
        lines = list(map(lambda e: [self.vertices[e.v(0)].point, self.vertices[e.v(1)].point], self.edges))
        lc = mc.LineCollection(lines, colors='k', linewidths=1, zorder=1)
        plt.gca().add_collection(lc)
        plt.plot()


def test():
    import numpy as np
    points = [
        [1.0, 1.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [5.0, 7.0],
        [9.0, 11.0],
        [10.0, 11.0],
        [-1.0, 8.0]
    ]
    plt.plot(*np.transpose(points), 'ok')
    f = Filtration(points)
    f.print()
