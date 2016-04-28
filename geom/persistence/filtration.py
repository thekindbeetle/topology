import math
import geom.all_vertices
import geom.all_edges
import geom.all_triangles
import geom.vert
from operator import attrgetter


class Filtration:
    """
    Фильтрация Чеха для данного множества вершин в R^2
    """

    # Вершины комплекса (AllVertices)
    vertices = None

    # Рёбра комплекса (AllEdges)
    edges = None

    # Треугольники комплекса (AllTriangles)
    triangles = None

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

        # Индексирование вершин, рёбер и треугольников
        self.vertices = geom.all_vertices.AllVertices([geom.vert.Vert(idx, points[idx][0], points[idx][1]) for idx in range(len(points))])
        self.triangles = geom.all_triangles.AllTriangles(points)
        self.edges = geom.all_edges.AllEdges(self.triangles)
        self.vertices.init_inc_edges(self.edges)
        self.vertices.init_inc_triangles(self.triangles)
        self.triangles.init_incident_edges(self.vertices, self.edges)
        self.edges.init_incident_triangles(self.triangles)
        self.edges.init_board_edges()
        self.vertices.init_board_vertices(self.edges)
        self.triangles.add_out(self.vertices, self.edges)

        self.vertNum = self.vertices.count()
        self.edgeNum = self.edges.count()
        self.trNum = self.triangles.count() # включая внешность
        simpNum = self.vertNum + self.edgeNum + self.trNum

        self.simplexes = []

        # Добавление вершин, ребер, треугольников, внешности
        for i in range(self.vertNum):
            self.simplexes.append(self.vertices.get_vert(i))
        for i in range(self.edgeNum):
            self.simplexes.append(self.edges.get_edge(i))
        for i in range(self.trNum):
            self.simplexes.append(self.triangles.get_triangle(i))

        # Инициализация времен появления
        for s in self.simplexes:
            s.set_appearance_time(self.vertices, self.edges, self.triangles)

        # Сортировка списка симплексов по времени появления
        self.sort_simplexes()

        # Инициализация индексов фильтрации симплексов
        for i in range(simpNum):
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
        tr_glob_indexes = self.edges.incident_triangles_of_edge(edge.globInd)
        global_tr_idx_0 = tr_glob_indexes[0]
        global_tr_idx_1 = tr_glob_indexes[1]
        filt_tr_idx_0 = self.triangles.get_triangle(global_tr_idx_0).filtInd
        filt_tr_idx_1 = self.triangles.get_triangle(global_tr_idx_1).filtInd
        return [filt_tr_idx_0, filt_tr_idx_1]

    def sort_simplexes(self):
        """
        Сортировка симплексов по времени появления.
        Важно! Используется устойчивая сортировка.
        Поскольку в исходном списке треугольники идут после рёбер,
        треугольники будут идти после рёбер с одинаковым временем появления.
        :return:
        """
        #print("Sort procedure starts...")
        self.simplexes.sort(key=attrgetter('appTime'))
        #print("Simplexes successfully sorted.")

    def print(self):
        print("Filtration")
        for s in self.simplexes:
            print("f.ind: {0}, appearance time = {1}, {2}".format(s.filtInd, s.appTime, s))

    def print_min_max(self):
        print("Minimal appearance time: {0}".format(self.get_min_app_time()))
        print("Maximal appearance time: {0}".format(self.get_max_app_time()))