import matplotlib.pyplot as pyplot
import generators.poisson
import geom.vert
import geom.all_vertices
import geom.all_edges
import geom.all_triangles
import geom.persistence.filtration
import geom.persistence.persistence
import triangle

def get_persistence_diagram(points):
    """
    Вычисление диаграммы персистентности для 1-циклов по набору точек.
    Строится фильтрация Чеха, по которой вычисляется диаграмма персистентности.
    :param points: список точек из R^2
    :return:
    """
    f = geom.persistence.filtration.Filtration(points)
    pers = geom.persistence.persistence.Persistence(f)
    return pers.get_diagram()


def test_easy():
    points = [[1.0, 2.0, 2.0, 5.0, 9.0, 10.0, -1.0], [1.0, 1.0, 2.0, 7.0, 11.0, 11.0, 8.0]]
    d = triangle.delaunay([(points[0][i], points[1][i]) for i in range(len(points[0]))])
    diagram = get_persistence_diagram([[points[0][i], points[1][i]] for i in range(len(points[0]))])
    print(diagram)
    pyplot.scatter(points[0], points[1], marker='o')
    pyplot.triplot(points[0], points[1], d)
    pyplot.show()

def test_hard():
    points = generators.poisson.poisson_homogeneous_point_process(1, 50)
    diagram = get_persistence_diagram(points)
    pyplot.scatter([diagram[i][0] for i in range(len(diagram))], [diagram[i][1] for i in range(len(diagram))], marker='o')
    pyplot.show()