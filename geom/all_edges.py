

class AllEdges:
    """
    Класс, хранящий все рёбра триангуляции
    """

    # список рёбер триангуляции
    edges = None

    # список списков инцидентных треугольников
    # ребру с индексом i соответствует список [i] инцидентных треугольников
    incidTriangles = None

    # список граничных рёбер триангуляции
    boardEdges = None

    def __init__(self, triangles):
        """
        Создание списка рёбер по списку треугольников триангуляции
        :param triangles: экземпляр AllTriangles
        """
        self.edges = triangles.get_all_edges()
        self.incidTriangles = [[] for i in range(len(self.edges))]

    def count(self):
        """
        Количество рёбер триангуляции
        :return:
        """
        return len(self.edges)

    def get_edge(self, idx):
        """
        Ребро триангуляции по индексу
        :param idx: глобальный индекс ребра
        :return:
        """
        return self.edges[idx]

    def get_0_vert_of_edge(self, idx):
        """
        Начальная вершина ребра с данным глобальным индексом
        :param idx: индекс ребра
        :return:
        """
        return self.edges[idx].v(0)

    def get_1_vert_of_edge(self, idx):
        """
        Конечная вершина ребра с данным глобальным индексом
        :param idx: индекс ребра
        :return:
        """
        return self.edges[idx].v(1)

    def incident_triangles_of_edge(self, idx):
        """
        Список инцидентных ребру треугольников
        :param idx: индекс треугольника
        :return:
        """
        return self.incidTriangles[idx]

    def add_incident_triangle(self, edge_list, triang_idx):
        """
        Добавить рёбрам из списка инцидентный треугольник с заданным индексом
        :param edge_list: список индексов рёбер
        :param triang_idx: индекс инцидентного треугольника
        :return:
        """
        for edge_idx in edge_list:
            self.incidTriangles[edge_idx].append(triang_idx)

    def init_incident_triangles(self, triangles):
        """
        Инициализация списка инцидентных треугольников
        :param triangles: экземпляр AllTriangles
        :return:
        """
        # Пробегаем все треугольни#ки, каждый записываем в 3 списка,
        # соответствующих ребрам этого треугольника.
        # Результат: массив, индексы которого - глобальные номера ребер,
        # в i-ой ячейке - список индексов треугольников, инцедентных i-ому ребру.
        for i in range(triangles.size()):
            self.incidTriangles[triangles.incident_edges_of_triangle(i)[0]].append(i)
            self.incidTriangles[triangles.incident_edges_of_triangle(i)[1]].append(i)
            self.incidTriangles[triangles.incident_edges_of_triangle(i)[2]].append(i)

    def init_board_edges(self):
        """
        Инициализация списка граничных рёбер
        :return:
        """
        self.boardEdges = []
        for edge_idx in range(self.count()):
            if(len(self.incidTriangles[edge_idx]) == 1):
                self.boardEdges.append(edge_idx)

    def print(self):
        print("Edges:")
        for e in self.edges:
            print(e)
        print("Incident triangles to edges:")
        for i in range(self.count()):
            print("e[{0}]".format(i))
            for tr in self.incidTriangles[i]:
                print(tr)
        print("Board edges: {0}".format(self.boardEdges))