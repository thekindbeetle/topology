import geom
import geom.vert
import geom.edge
import geom.triang
import triangle


class AllTriangles:
    """
    Класс, хранящий все треугольники триангуляции (плюс внешнюю грань)
    """

    # список треугольников триангуляции
    triangles = None

    # список списков инцидентных рёбер
    incidEdges = None
    def __init__(self, vertices):
        tr = triangle.delaunay(vertices)
        # Получение треугольников триангуляции
        self.triangles = [geom.triang.Triang(idx, tr[idx][0], tr[idx][1], tr[idx][2]) for idx in range(len(tr))]
        # Создание пустого массива списков ребер, инцидентных треугольникам
        self.incidEdges = [[] for i in range(len(self.triangles))]

        ##!! TODO: добавить внешность

    def size(self):
        """
        Количество треугольников в триангуляции в текущий момент.
        (вне зависимости от того, добавлена внешняя грань или нет)
        :return:
        """
        return len(self.triangles)

    def get_triangle(self, triang_idx):
        return self.triangles[triang_idx]

    def a_vert_of_triang(self, triang_idx):
        return self.triangles[triang_idx].v(0)

    def b_vert_of_triang(self, triang_idx):
        return self.triangles[triang_idx].v(1)

    def c_vert_of_triang(self, triang_idx):
        return self.triangles[triang_idx].v(2)

    def incident_edges_of_triangle(self, triang_idx):
        return self.incidEdges[triang_idx]

    def get_all_edges(self):
        edges = [] # список рёбер
        idx = 0

        # внешность не учитываем
        for triang_idx in range(self.size()):
            tr = self.triangles[triang_idx]
            if not geom.edge.Edge.contains_edge(edges, tr.v(0), tr.v(1)):
                edges.append(geom.edge.Edge(idx, tr.v(0), tr.v(1)))
                idx += 1
            if not geom.edge.Edge.contains_edge(edges, tr.v(0), tr.v(2)):
                edges.append(geom.edge.Edge(idx, tr.v(0), tr.v(2)))
                idx += 1
            if not geom.edge.Edge.contains_edge(edges, tr.v(2), tr.v(1)):
                edges.append(geom.edge.Edge(idx, tr.v(2), tr.v(1)))
                idx += 1
        return edges

    def add_out(self, vertices, edges):
        """
        Add outer face to the triangulation
        :param vertices: AllVertices instance
        :param edges: AllEdges instance
        :return:
        """
        outIdx = self.size()
        out = geom.triang.Out(outIdx, vertices.boardVertices)
        self.triangles.append(out)
        self.incidEdges.append(edges.boardEdges)
        vertices.add_inc_triangle(vertices.boardVertices, outIdx)
        edges.add_incident_triangle(edges.boardEdges, outIdx)


    def init_incident_edges(self, vertices, edges):
        """
        Инициализация списков инцидентных рёбер
        :param vertices: экземпляр AllVertices
        :param edges: экземпляр AllEdges
        :return:
        """
        # Пробегаем все треугольники
        for i in range(self.size()):
            # Количество ребер, инцидентных вершине A треугольника i.
            iAEdgesCount = vertices.count_of_inc_edges_of_vert(self.a_vert_of_triang(i))
            # Для i-ого треугольника просматриваем список инцедентных ребер вершины A.
            # Если текущее (j-ое) ребро равно ребру AB или AC i-ого треугольника,
            # добавляем это ребро в список ребер, инцедентных i-ому треугольнику.
            for j in range(iAEdgesCount):
                # Глобальный индекс j-ого ребра инцидентного вершине A i-ого треугольника.
                iAj = vertices.inc_edges_of_vert(self.a_vert_of_triang(i))[j]
                if(edges.get_edge(iAj).equals(self.a_vert_of_triang(i), self.b_vert_of_triang(i)) or
                   edges.get_edge(iAj).equals(self.a_vert_of_triang(i), self.c_vert_of_triang(i))):
                    self.incidEdges[i].append(iAj)
            # Количество ребер, инцидентных вершине B треугольника i.
            iBEdgesCount = vertices.count_of_inc_edges_of_vert(self.b_vert_of_triang(i))#
            # Для i-ого треугольника просматриваем список инцедентных ребер вершины B.
            # Если текущее (j-ое) ребро равно ребру BC i-ого треугольника,
            # добавляем это ребро в список ребер, инцедентных i-ому треугольнику.
            for j in range(iBEdgesCount):
                # Глобальный индекс j-ого ребра инцидентного вершине B i-ого треугольника.
                iBj = vertices.inc_edges_of_vert(self.b_vert_of_triang(i))[j]
                if(edges.get_edge(iBj).equals(self.b_vert_of_triang(i),self.c_vert_of_triang(i))):
                    self.incidEdges[i].append(iBj)

    def print(self):
        print("Triangles:")
        for tr in self.triangles:
            print(tr)
        print("Incident edges to triangles:")
        for i in range(self.size()):
            print("t[{0}]:".format(i))
            for e in self.incidEdges[i]:
                print(e)

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
    # verts.append(geom.vert.Vert(0, 1.0, 1.0))
    # verts.append(geom.vert.Vert(1, 2.0, 1.0))
    # verts.append(geom.vert.Vert(2, 2.0, 2.0))
    # verts.append(geom.vert.Vert(3, 5.0, 7.0))
    # verts.append(geom.vert.Vert(4, 9.0, 11.0))
    # verts.append(geom.vert.Vert(5, 10.0, 11.0))
    # verts.append(geom.vert.Vert(6, -1.0, 8.0))
    AllTriangles(verts).print()