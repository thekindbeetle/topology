

class AllVertices:
    """
    Класс, хранящий все вершины триангуляции
    """

    # список вершин (Vert)
    vertices = None

    # список списков инцидентных рёбер
    # вершине с индексом i соответствует список [i] инцидентных рёбер
    incidEdges = None

    # список списков инцидентных треугольников
    # вершине с индексом i соответствует список [i] инцидентных треугольников
    incidTriangles = None

    # список граничных вершин триангуляции
    boardVertices = None

    def __init__(self, vertices):
        """
        :param vertices: список вершин (Vert)
        """
        self.vertices = vertices
        print("Vertices count: {0}".format(self.count()))
        self.incidEdges = [[] for i in range(len(vertices))]
        self.incidTriangles = [[] for i in range(len(vertices))]

    def count(self):
        """
        Количество вершин
        :return:
        """
        return len(self.vertices)

    def get_vert(self, idx):
        """
        Вершина с заданным индексом
        :param idx: индекс вершины
        :return:
        """
        return self.vertices[idx]

    def inc_edges_of_vert(self, idx):
        """
        Список инцидентных вершине рёбер
        :param idx: индекс вершины
        :return:
        """
        return self.incidEdges[idx]

    def inc_triangles_of_vert(self, idx):
        """
        Список инцидентных вершине треугольников
        :param idx: индекс вершины
        :return:
        """
        return self.incidTriangles[idx]

    def count_of_inc_edges_of_vert(self, idx):
        """
        Количество инцидентных вершине рёбер
        :param idx: индекс вершины
        :return:
        """
        return len(self.incidEdges[idx])

    def count_of_inc_triangles_of_vert(self, idx):
        """
        Количество инцидентных вершине рёбер
        :param idx: индекс вершины
        :return:
        """
        return len(self.incidTriangles[idx])

    def add_inc_triangle(self, vert_list, triang_idx):
        """
        Записать треугольник в качестве инцидентного набору вершин
        :param vert_list: набор индексов вершин
        :param triang_idx: набор треугольников
        :return:
        """
        for vert_idx in vert_list:
            self.incidTriangles[vert_idx].append(triang_idx)

    def init_inc_edges(self, edges):
        """
        Инициализация списков инцидентных рёбер
        :param edges: экземпляр AllEdges
        :return:
        """

        # Пробегаем все ребра, каждое приписываем двум его вершинам.
        # Результат: массив, индексы которого - глоальные номера точек,
        # в i-ой ячейке - список индексов ребер, инцедентных i-ой вершине.
        for idx in range(edges.count()):
            v0 = edges.get_0_vert_of_edge(idx);
            v1 = edges.get_1_vert_of_edge(idx);
            self.incidEdges[v0].append(idx)
            self.incidEdges[v1].append(idx)

    def init_inc_triangles(self, triangles):
        """
        Инициализация списков инцидентных треугольников
        :param triangles: экземпляр AllTriangles#
        :return:
        """
        # Пробегаем все треугольники, каждый записываем в 3 списка,
        # соответствующих вершинам этого треугольника.
        # Результат: массив, индексы которого - глоальные номера точек,
        # в i-й ячейке - список индексов треугольников, содержащих i-ую вершину.
        for idx in range(triangles.count()):
            self.incidTriangles[triangles.a_vert_of_triang(idx)].append(idx)
            self.incidTriangles[triangles.b_vert_of_triang(idx)].append(idx)
            self.incidTriangles[triangles.c_vert_of_triang(idx)].append(idx)

    def init_board_vertices(self, edges):
        """
        Инициализация списка граничных вершин
        :param edges: экземпляр AllEdges
        :return:
        """
        self.boardVertices = []
        for edge_idx in edges.boardEdges:
            v0 = edges.get_edge(edge_idx).v(0)
            v1 = edges.get_edge(edge_idx).v(1)
            if v0 not in self.boardVertices:
                self.boardVertices.append(v0)
            if v1 not in self.boardVertices:
                self.boardVertices.append(v1)

    def print( self ):
        print("Vertices:")
        for v in self.vertices:
            v.print()

        print("Incedent edges to vertices:")
        for vert_idx in range(self.count()):
            print("v[{0}]".format(vert_idx))
            for edge in self.incidEdges[vert_idx]:
                edge.print()
        print("Incedent triangles to vertices:")
        for vert_idx in range(self.count()):
            print("v[{0}]".format(vert_idx))
            for triangle in self.incidTriangles[vert_idx]:
                triangle.print()
        print("Board vertices:")
        for board_vert in self.boardVertices:
            board_vert.print()