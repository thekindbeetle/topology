import matplotlib.pyplot as pyplot
from scipy.stats import *
import geom.vert
import geom.all_vertices
import geom.all_edges
import geom.all_triangles
import geom.persistence.filtration
import geom.persistence.persistence
import triangle


def poisson_point_process(rate, Dx, Dy=None):
    """
    Determines the number of events `N` for a rectangular region,
    given the rate `rate` and the dimensions, `Dx`, `Dy`.
    Returns a <2xN> NumPy array.
    """
    if Dy is None: Dy = Dx
    N = poisson(rate * Dx * Dy).rvs()
    x = uniform.rvs(0, Dx, N)
    y = uniform.rvs(0, Dy, N)
    out = [[x[i], y[i]] for i in range(N)]
    print("{0} Poisson points generated.".format(N))
    return out


def get_persistence_diagram(points):
    """
    Вычисление диаграммы персистентности для 1-циклов по набору точек.
    Строится фильтрация Чеха, по которой вычисляется диаграмма персистентности.
    :param points: список точек из R^2
    :return:
    """
    vertices = geom.all_vertices.AllVertices([geom.vert.Vert(idx, points[idx][0], points[idx][1]) for idx in range(len(points))])
    triangles = geom.all_triangles.AllTriangles(points)
    edges = geom.all_edges.AllEdges(triangles)
    vertices.init_inc_edges(edges)
    vertices.init_inc_triangles(triangles)
    triangles.init_incident_edges(vertices, edges)
    edges.init_incident_triangles(triangles)
    edges.init_board_edges()
    vertices.init_board_vertices(edges)
    triangles.add_out(vertices, edges)
    f = geom.persistence.filtration.Filtration(vertices, edges, triangles)
    pers = geom.persistence.persistence.Persistence(f)
    return pers.get_diagram()


def test():
    lamb = 100
    # points = PoissonPP(lamb, 1)
    points = [[1.0, 2.0, 2.0, 5.0, 9.0, 10.0, -1.0], [1.0, 1.0, 2.0, 7.0, 11.0, 11.0, 8.0]]
    d = triangle.delaunay([(points[0][i], points[1][i]) for i in range(len(points[0]))])
    pyplot.scatter(points[0], points[1], marker='o')
    pyplot.triplot(points[0], points[1], d)
    pyplot.show()


points = poisson_point_process(1, 30)
diagram = get_persistence_diagram(points)
pyplot.scatter([diagram[i][0] for i in range(len(diagram))], [diagram[i][1] for i in range(len(diagram))], marker='o')
pyplot.show()