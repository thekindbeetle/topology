import math
import numpy as np
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

    def __init__(self):
        self.vertices = []
        self.edges = []
        self.triangles = []
        self.incidEdgesToVertices = []
        self.incidTrianglesToVertices = []
        self.boardVertices = []
        self.incidTrianglesToEdges = []
        self.boardEdges = []
        self.incidEdgesToTriangles = []
        self.simplexes = []

    @classmethod
    def from_points(cls, x, y):
        """
        Построение фильтрации Чеха по набору точек из R^2.
        :param points:
        :return:
        """
        if len(x) != len(y):
            raise RuntimeError('Длины x и y должны совпадать!')
        f = Filtration()
        f.vertNum = len(x)

        # Индексирование вершин, рёбер и треугольников
        f.vertices = [geom.vert.Vert(idx, x[idx], y[idx]) for idx in range(f.vertNum)]
        f.incidEdgesToVertices = [[] for i in range(f.vertNum)]
        f.incidTrianglesToVertices = [[] for i in range(f.vertNum)]

        tr = triangle.delaunay(np.transpose([x, y]))
        f.trNum = len(tr)

        f.triangles = [geom.triang.Triang(idx, tr[idx][0], tr[idx][1], tr[idx][2]) for idx in range(f.trNum)]
        f.incidEdgesToTriangles = [[] for i in range(f.trNum)]

        # Индексируем рёбра для быстрого поиска
        edges_idx = dict()

        idx = 0

        # внешность не учитываем
        for triIdx in range(f.trNum):
            tr = f.triangles[triIdx]
            if not edges_idx.get((tr.v(0), tr.v(1))):
                f.edges.append(geom.edge.Edge(idx, tr.v(0), tr.v(1)))
                edges_idx[(tr.v(0), tr.v(1))] = True
                edges_idx[(tr.v(1), tr.v(0))] = True
                idx += 1
            if not edges_idx.get((tr.v(0), tr.v(2))):
                f.edges.append(geom.edge.Edge(idx, tr.v(0), tr.v(2)))
                edges_idx[(tr.v(0), tr.v(2))] = True
                edges_idx[(tr.v(2), tr.v(0))] = True
                idx += 1
            if not edges_idx.get((tr.v(1), tr.v(2))):
                f.edges.append(geom.edge.Edge(idx, tr.v(1), tr.v(2)))
                edges_idx[(tr.v(1), tr.v(2))] = True
                edges_idx[(tr.v(2), tr.v(1))] = True
                idx += 1

        f.edgeNum = len(f.edges)
        f.incidTrianglesToEdges = [[] for i in range(f.edgeNum)]

        # Инициализация списков инцидентных рёбер для вершин
        # Пробегаем все ребра, каждое приписываем двум его вершинам.
        # Результат: массив, индексы которого - глоальные номера точек,
        # в i-ой ячейке - список индексов ребер, инцедентных i-ой вершине.
        for idx in range(f.edgeNum):
            v0 = f.edges[idx].v(0)
            v1 = f.edges[idx].v(1)
            f.incidEdgesToVertices[v0].append(idx)
            f.incidEdgesToVertices[v1].append(idx)

        # Инициализация списков инцидентных треугольников для вершин
        # Пробегаем все треугольники, каждый записываем в 3 списка,
        # соответствующих вершинам этого треугольника.
        # Результат: массив, индексы которого - глоальные номера точек,
        # в i-й ячейке - список индексов треугольников, содержащих i-ую вершину.
        for idx in range(f.trNum):
            f.incidTrianglesToVertices[f.triangles[idx].v(0)].append(idx)
            f.incidTrianglesToVertices[f.triangles[idx].v(1)].append(idx)
            f.incidTrianglesToVertices[f.triangles[idx].v(2)].append(idx)

        # Инициализация списков инцидентных рёбер для треугольников
        # Пробегаем все треугольники
        for i in range(f.trNum):
            # Количество ребер, инцидентных вершине A треугольника i.
            iAEdgesCount = len(f.incidEdgesToVertices[f.triangles[i].v(0)])
            # Для i-ого треугольника просматриваем список инцедентных ребер вершины A.
            # Если текущее (j-ое) ребро равно ребру AB или AC i-ого треугольника,
            # добавляем это ребро в список ребер, инцедентных i-ому треугольнику.
            for j in range(iAEdgesCount):
                # Глобальный индекс j-ого ребра инцидентного вершине A i-ого треугольника.
                iAj = f.incidEdgesToVertices[f.triangles[i].v(0)][j]
                if(f.edges[iAj].equals(f.triangles[i].v(0), f.triangles[i].v(1)) or
                   f.edges[iAj].equals(f.triangles[i].v(0), f.triangles[i].v(2))):
                    f.incidEdgesToTriangles[i].append(iAj)
            # Количество ребер, инцидентных вершине B треугольника i.
            iBEdgesCount = len(f.incidEdgesToVertices[f.triangles[i].v(1)])
            # Для i-ого треугольника просматриваем список инцедентных ребер вершины B.
            # Если текущее (j-ое) ребро равно ребру BC i-ого треугольника,
            # добавляем это ребро в список ребер, инцедентных i-ому треугольнику.
            for j in range(iBEdgesCount):
                # Глобальный индекс j-ого ребра инцидентного вершине B i-ого треугольника.
                iBj = f.incidEdgesToVertices[f.triangles[i].v(1)][j]
                if f.edges[iBj].equals(f.triangles[i].v(1), f.triangles[i].v(2)):
                    f.incidEdgesToTriangles[i].append(iBj)

        # Инициализация списка инцидентных треугольников для рёбер
        # Пробегаем все треугольники, каждый записываем в 3 списка,
        # соответствующих ребрам этого треугольника.
        # Результат: массив, индексы которого - глобальные номера ребер,
        # в i-ой ячейке - список индексов треугольников, инцедентных i-ому ребру.
        for i in range(f.trNum):
            f.incidTrianglesToEdges[f.incidEdgesToTriangles[i][0]].append(i)
            f.incidTrianglesToEdges[f.incidEdgesToTriangles[i][1]].append(i)
            f.incidTrianglesToEdges[f.incidEdgesToTriangles[i][2]].append(i)

        # Инициализация списка граничных рёбер
        for edge_idx in range(f.edgeNum):
            if len(f.incidTrianglesToEdges[edge_idx]) == 1:
                f.boardEdges.append(edge_idx)

        # Инициализация списка граничных вершин
        for edge_idx in f.boardEdges:
            v0 = f.edges[edge_idx].v(0)
            v1 = f.edges[edge_idx].v(1)
            if v0 not in f.boardVertices:
                f.boardVertices.append(v0)
            if v1 not in f.boardVertices:
                f.boardVertices.append(v1)

        # Add outer face to the triangulation
        outIdx = f.trNum
        out = geom.triang.Out(outIdx, f.boardVertices)
        f.triangles.append(out)
        f.incidEdgesToTriangles.append(f.boardEdges)

        for vert_idx in f.boardVertices:
            f.incidTrianglesToVertices[vert_idx].append(outIdx)

        for edge_idx in f.boardEdges:
            f.incidTrianglesToEdges[edge_idx].append(outIdx)

        f.trNum += 1  # учитываем внешность

        f.simpNum = f.vertNum + f.edgeNum + f.trNum

        # Добавление вершин, ребер, треугольников, внешности
        for i in range(f.vertNum):
            f.simplexes.append(f.vertices[i])
        for i in range(f.edgeNum):
            f.simplexes.append(f.edges[i])
        for i in range(f.trNum):
            f.simplexes.append(f.triangles[i])

        # Инициализация времен появления
        for s in f.simplexes:
            if s.dim == 0:
                s.appTime = 0
            elif s.dim == 1:
                length = geom.util.dist(f.vertices[s.v(0)], f.vertices[s.v(1)])
                s.appTime = length / 2
            elif s.dim == 2:
                len_a = geom.util.dist(f.vertices[s.v(0)], f.vertices[s.v(1)])
                len_b = geom.util.dist(f.vertices[s.v(0)], f.vertices[s.v(2)])
                len_c = geom.util.dist(f.vertices[s.v(1)], f.vertices[s.v(2)])
                if geom.util.is_obtuse(len_a, len_b, len_c):
                    s.appTime = max([len_a, len_b, len_c]) / 2
                else:
                    s.appTime = geom.util.outer_radius(f.vertices[s.v(0)],
                                                       f.vertices[s.v(1)],
                                                       f.vertices[s.v(2)])
        f.simplexes[-1].appTime = max([f.triangles[i].appTime for i in range(f.trNum - 1)]) + 1

        # Сортировка списка симплексов по времени появления
        f.sort_simplexes()

        # Инициализация индексов фильтрации симплексов
        for i in range(f.simpNum):
            f.simplexes[i].filtInd = i

        return f
        
    @classmethod
    def from_grid(cls, values):
        """
        Построение фильтрации по скалярному полю на плоскости.
        :param values: NumPy Array
        :return:
        """
        f = Filtration()
        lx, ly = values.shape

        f.vertNum = lx * ly
        f.edgeNum = 3 * lx * ly - 2 * (lx + ly) + 1
        f.trNum = 2 * (lx - 1) * (ly - 1)

        horizEdgeNum = lx * (ly - 1)
        verticalEdgeNum = ly * (lx - 1)
        diagEdgeNum = (lx - 1) * (ly - 1)

        for i in range(lx):
            for j in range(ly):
                v = geom.vert.Vert(ly * i + j, i, j)
                v.appTime = values[i, j]
                f.vertices.append(v)
                # Горизонтальные рёбра
                if j != 0:
                    e = geom.edge.Edge(v.globInd - i, v.globInd - 1, v.globInd)
                    e.appTime = max(f.vertices[v.globInd - 1].appTime, v.appTime)
                    f.edges.append(e)
                # Вертикальные рёбра
                if i != 0:
                    e = geom.edge.Edge(horizEdgeNum + v.globInd - ly, v.globInd - ly, v.globInd)
                    e.appTime = max(f.vertices[v.globInd - ly].appTime, v.appTime)
                    f.edges.append(e)
                # Диагональные рёбра
                if i != 0 and j != 0:
                    e = geom.edge.Edge(horizEdgeNum + verticalEdgeNum + v.globInd - ly - i,
                                       v.globInd - ly - 1, v.globInd)
                    e.appTime = max(f.vertices[v.globInd - ly - 1].appTime, v.appTime)
                    f.edges.append(e)

        trCounter = 0

        # Правые треугольники
        for i in range(lx - 1):
            for j in range(ly - 1):
                t = geom.triang.Triang(trCounter, ly * i + j, ly * i + j + 1, ly * (i + 1) + j + 1)
                t.appTime = max(f.vertices[ly * i + j].appTime,
                                f.vertices[ly * i + j + 1].appTime,
                                f.vertices[ly * (i + 1) + j + 1].appTime)
                f.incidEdgesToTriangles.append([])
                trCounter += 1

        # Левые треугольники
        for i in range(lx - 1):
            for j in range(ly - 1):
                t = geom.triang.Triang(trCounter, ly * i + j, ly * (i + 1) + j, ly * (i + 1) + j + 1)
                t.appTime = max(f.vertices[ly * i + j].appTime,
                                f.vertices[ly * (i + 1) + j].appTime,
                                f.vertices[ly * (i + 1) + j + 1].appTime)
                trCounter += 1

        f.simplexes = f.vertices + f.edges + f.triangles
        return f

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
    # points = [
    #     [1.0, 1.0],
    #     [2.0, 1.0],
    #     [2.0, 2.0],
    #     [5.0, 7.0],
    #     [9.0, 11.0],
    #     [10.0, 11.0],
    #     [-1.0, 8.0]
    # ]
    points = [
        [1.0, 1.0, 8.0],
        [2.0, 1.0, 4.0],
        [2.0, 2.0, 2.0],
        [5.0, 7.0, 13.0],
        [9.0, 11.0, 8.0],
        [10.0, 11.0, -1.0],
        [-1.0, 8.0, -5.0]
    ]
    plt.matshow(points)
    # plt.plot(*np.transpose(points), 'ok')
    # f = Filtration.from_points(points)
    f = Filtration.from_grid(np.asarray(points))
    # print(f.incidEdgesToTriangles)
    # print(f.incidTrianglesToEdges)
    f.print()
    f.draw()
    plt.show()

