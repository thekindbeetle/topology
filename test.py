import matplotlib.pyplot as pyplot
import generators.poisson
import generators.matern
import generators.thomas
import geom.vert
import geom.all_vertices
import geom.all_edges
import geom.all_triangles
import geom.persistence.filtration
import geom.persistence.persistence
import triangle
import numpy as np


def get_persistence(points):
    """
    Вычисление диаграммы персистентности для 1-циклов по набору точек.
    Строится фильтрация Чеха, по которой вычисляется диаграмма персистентности.
    :param points: список точек из R^2
    :return:
    """
    f = geom.persistence.filtration.Filtration(points)
    pers = geom.persistence.persistence.Persistence(f)
    return pers


def test_easy():
    points = [[1.0, 2.0, 2.0, 5.0, 9.0, 10.0, -1.0], [1.0, 1.0, 2.0, 7.0, 11.0, 11.0, 8.0]]
    d = triangle.delaunay([(points[0][i], points[1][i]) for i in range(len(points[0]))])
    diagram = get_persistence([[points[0][i], points[1][i]] for i in range(len(points[0]))]).get_diagram()
    print(diagram)
    pyplot.scatter(points[0], points[1], marker='o')
    pyplot.triplot(points[0], points[1], d)
    pyplot.show()


def test_hard():
    points = generators.poisson.poisson_homogeneous_point_process(1, 50)
    d = triangle.delaunay(points)
    diagram = get_persistence(points).get_diagram()
    pyplot.figure(1)
    pyplot.triplot([points[i][0] for i in range(len(points))], [points[i][1] for i in range(len(points))], d)
    pyplot.figure(2)
    pyplot.scatter([diagram[i][0] for i in range(len(diagram))], [diagram[i][1] for i in range(len(diagram))], marker='o')
    pyplot.show()

tim = []
for i in range(2):
    points = generators.poisson.poisson_homogeneous_point_process(1, 5000)
    d = triangle.delaunay(points)
    pers = get_persistence(points)
    #pyplot.figure(1)
    pyplot.plot(range(len(pers.numOfComponents)), pers.numOfComponents, '-b',
                range(len(pers.numOfBigComponents)), pers.numOfBigComponents, '-r',
                range(len(pers.numOfCycles)), pers.numOfCycles, '-g',)
    #pyplot.show()
    b_1 = max(pers.numOfCycles)
    n_a = max(pers.numOfBigComponents)
    tim.append(b_1 / n_a)

print(np.mean(tim))
print(np.sqrt(np.var(tim)))
tim.clear()

for i in range(2):
    points = generators.matern.matern_point_process(280, 0.05, 17, 1)
    print(len(points))
    d = triangle.delaunay(points)
    pers = get_persistence(points)
    #pyplot.figure(1)
    pyplot.plot(range(len(pers.numOfComponents)), pers.numOfComponents, '--b',
                range(len(pers.numOfBigComponents)), pers.numOfBigComponents, '--r',
                range(len(pers.numOfCycles)), pers.numOfCycles, '--g',)
    #pyplot.show()
    b_1 = max(pers.numOfCycles)
    n_a = max(pers.numOfBigComponents)
    tim.append(b_1 / n_a)


pyplot.show()
print(np.mean(tim))
print(np.sqrt(np.var(tim)))