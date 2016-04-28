import math
import geom.util


class Triang:
    v0 = None
    v1 = None
    v2 = None

    # Appearance time of triangle:
    # for oxygon - circumscribed circle;
    # for obtuse triangle - half largest side;
    # for out - maximum appearance time of triangle + 1
    appTime = None

    # Global index, depends on dimension:
    # for 0-simplex - index in list of all vertices;
    # for 1-simplex - index in list of all edges;
    # for 2-simplex - index in triangulation
    globInd = None

    # Index in filtration
    filtInd = None

    dim = 2

    def __init__(self, idx, v0=None, v1=None, v2=None):
        """
        :param idx: Глобальный индекс треугольника
        :param v0: индекс первой вершины
        :param v1: индекс второй вершины
        :param v2: индекс третьей вершины
        """
        self.globInd = idx
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    def __repr__(self):
        return "Triangle #{0}, [{1}, {2}, {3}]".format(self.globInd, self.v0, self.v1, self.v2)

    def edge0( self, edges, triangs ):
        edge_idx = triangs.incident_edges_of_triangle(self.globInd)[0]
        return edges.get_edge(edge_idx)

    def edge1( self, edges, triangs ):
        edge_idx = triangs.incident_edges_of_triangle(self.globInd)[1]
        return edges.get_edge(edge_idx)

    def edge2( self, edges, triangs ):
        edge_idx = triangs.incident_edges_of_triangle(self.globInd)[2]
        return edges.get_edge(edge_idx)

    def edges( self, e, t ):
        res = []
        edges = t.incedentEdgesOfTriangle(self.globInd)
        for i in range(len(e)) :
            res.append(e.get_edge(edges[i]))
        return res

    def get_triang( self, vertices ):
        a = vertices.get_vert(self.v0).point
        b = vertices.get_vert(self.v1).point
        c = vertices.get_vert(self.v2).point
        return [a, b, c]

    def outer_radius( self, vertices ):
        tr = self.get_triang(vertices)
        return geom.util.outer_radius(tr[0], tr[1], tr[2])

    def set_appearance_time( self, vertices, edges, triangles ):
        """
        Установить время появления симплекса в фильтрации
        :param vertices:
        :param edges:
        :param triangles:
        :return:
        """
        # длины сторон треугольника
        a = self.edge0(edges, triangles).get_length(vertices)
        b = self.edge1(edges, triangles).get_length(vertices)
        c = self.edge2(edges, triangles).get_length(vertices)

        if geom.util.is_obtuse(a, b, c):
            self.appTime = max(a, b, c) / 2
        else:
            self.appTime = geom.util.outer_radius_by_sides(a, b, c)

    def v( self, idx ):
        """
        Вершина треугольника по индексу 0..2
        :param idx: индекс вершины в треугольнике
        :return:
        """
        if idx == 0:
            return self.v0
        elif idx == 1:
            return self.v1
        elif idx == 2:
            return self.v2

    def equals_by_global_idx( self, simplex ):
        """
        Сравнение с другим симплексом по глобальному индексу.
        :param simplex: двугой симплекс
        :return:
        """
        return simplex.globInd == self.globInd

    def compare_to( self, simplex ):
        """
        Сравнить с другим симплексом по времени появления в фильтрации
        :param simplex:
        :return:
        """
        return math.copysign(1, self.appTime - simplex.appTime)


class Out(Triang):
    """
    Класс внешней грани триангуляции.
    """

    # Список вершин внешней грани
    verts = None

    def __init__(self, idx, verts=None):
        """
        Инициализация внешней грани
        :param idx: глобальный индекс в триангуляции
        :param verts: список вершин
        :return:
        """
        Triang.__init__(self, idx)
        self.verts = verts

    def __repr__(self):
        return "Outer face #{0}: {1}".format(self.globInd, self.verts)

    def set_appearance_time( self, vertices, edges, triangles ):
        """
        Устанавливаем внешности максимальное время появления
        :param vertices:
        :param edges:
        :param triangles:
        :return:
        """
        self.appTime = max([triangles.get_triangle(i).appTime for i in range(triangles.count() - 1)]) + 1 # исключая внешность

    def v( self, idx ):
        return self.verts[idx]

def test():
    a1 = 8
    b1 = 7
    c1 = 5
    a2 = 3
    b2 = 6
    c2 = 4
    t1 = Triang( 1, a1, b1, c1 )
    t2 = Triang( 2, a2, b2, c2 )
    print(t1)
    print(t2)
    o = Out( 10, [a1, a2, b1, b2, c1, c2])
    print(o)
    print(o.v(4))
    print(t1.compare_to(t1))
    print(t1.compare_to(t2))
    print(t1.equals_by_global_idx(t1))
    print(t1.equals_by_global_idx(t2))