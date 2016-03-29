import math
import geom.all_vertices
import geom.all_edges
import geom.all_triangles
import geom.vert

class Filtration:
    """
    Фильтрация Чеха для данного множества вершин в R^2
    """

    # Список симплексов фильтрации
    simplexes = None

    # Список времён появления симплексов в фильтрации
    times = None

    # Количество вершин
    vertNum = None

    # Количество рёбер
    edgeNum = None

    # Количество треугольников
    trNum = None

    def __init__(self, vertices, edges, triangles):
        self.vertNum = vertices.count()
        self.edgeNum = edges.count()
        self.trNum = triangles.size() # включая внешность
        simpNum = self.vertNum + self.edgeNum + self.trNum

        self.simplexes = []
        self.times = []

        # Добавление вершин, ребер, треугольников, внешности
        for i in range(self.vertNum):
            self.simplexes.append(vertices.get_vert(i))
        for i in range(self.edgeNum):
            self.simplexes.append(edges.get_edge(i))
        for i in range(self.trNum):
            self.simplexes.append(triangles.get_triangle(i))

        # Инициализация времен появления
        self.set_appearance_times(vertices, edges, triangles)

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

    def get_inc_triang_of_edge(self, edge_filt_idx, edges, triangles):
        edge = self.get_simplex(edge_filt_idx)
        tr_glob_indexes = edges.incident_triangles_of_edge(edge.globInd)
        global_tr_idx_0 = tr_glob_indexes[0]
        global_tr_idx_1 = tr_glob_indexes[1]
        filt_tr_idx_0 = triangles.get_triangle(global_tr_idx_0).filtInd
        filt_tr_idx_1 = triangles.get_triangle(global_tr_idx_1).filtInd
        return [filt_tr_idx_0, filt_tr_idx_1]

    def set_appearance_times(self, vertices, edges, triangles):
        for s in self.simplexes:
            s.set_appearance_time(vertices, edges, triangles)
            self.times.append(s.appTime)

    def sort_simplexes(self):
        eps = 0.0001 # погрешность вычисления
        # Сортируем одновременно список симплексов и список времён появления
        for i in range(self.simplexes_num()):
            for j in range(i):
                # Время появления j-го симплекса больше i-го, или их времена появления совпадают и один из симплексов - ребро, а другой - треугольник
                if ((self.times[j] > self.times[i] + eps) or (math.fabs(self.times[j] - self.times[i]) < eps and self.simplexes[j].dim == 2 and self.simplexes[i].dim == 1)):
                    self.times[i], self.times[j] = self.times[j], self.times[i]  # меняем местами пару симплексов и времена их появления
                    self.simplexes[i], self.simplexes[j] = self.simplexes[j], self.simplexes[i]

    def print(self):
        print("Filtratiion")
        for s in self.simplexes:
            print("f.ind: {0}, appearance time = {1}, {2}".format(s.filtInd, s.appTime, s))

    def print_min_max(self):
        print("Minimal appearance time: {0}".format(self.get_min_app_time()))
        print("Maximal appearance time: {0}".format(self.get_max_app_time()))

def test():
    verts = [
        [1.0, 1.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [5.0, 7.0],
        [9.0, 11.0],
        [10.0, 11.0],
        [-1.0, 8.0]
    ]
    vertices = geom.all_vertices.AllVertices([geom.vert.Vert(idx, verts[idx][0], verts[idx][1]) for idx in range(len(verts))])
    triangles = geom.all_triangles.AllTriangles(verts)
    edges = geom.all_edges.AllEdges(triangles)
    vertices.init_inc_edges(edges)
    vertices.init_inc_triangles(triangles)
    triangles.init_incident_edges(vertices, edges)
    edges.init_incident_triangles(triangles)
    edges.init_board_edges()
    vertices.init_board_vertices(edges)
    triangles.add_out(vertices, edges)
    f = Filtration(vertices, edges, triangles)
    f.print()

test()