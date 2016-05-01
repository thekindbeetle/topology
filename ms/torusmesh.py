import matplotlib.pyplot as plt
from ms.mesh import GridMesh
import numpy as np
import heapq


class TorusMesh(GridMesh):
    """
    Квадратная сетка на торе
    """

    # Список критических клеток
    cr_cells = []

    # Индикатор критических клеток
    cr_id = []

    # Дискретный градиент
    V = []

    def __init__(self, lx, ly):
        GridMesh.__init__(self, lx, ly)
        self.cr_id = np.zeros(4 * self.size, dtype=bool)
        self.V = [-1] * (4 * self.size)

    def critical_points(self, morse_index=None):
        """
        Вывести координаты критических точек с данным индексом.
        Если morse_index=None, то вывести весь список.
        :param morse_index: Индекс Морса критической точки
        :return: список координат X, список координат Y
        """
        if morse_index is None:
            x = [self.coords(c)[0] for c in self.cr_cells]
            y = [self.coords(c)[1] for c in self.cr_cells]
        else:
            x = [self.coords(c)[0] for c in self.cr_cells if self.dim(c) == morse_index]
            y = [self.coords(c)[1] for c in self.cr_cells if self.dim(c) == morse_index]
        return x, y

    def to_index(self, x, y):
        """
        Глобальный индекс вершины по координатам сетки.
        """
        return x * self.sizeY + y

    def coords(self, idx):
        """
        Координаты центра клетки (вершины, ребра или ячейки)
        """
        return np.mean([self.coordx(v) for v in self.verts(idx)]), np.mean([self.coordy(v) for v in self.verts(idx)])

    def dim(self, idx):
        """
        Размерность клетки
        """
        if idx < self.size:
            return 0
        elif idx < 3 * self.size:
            return 1
        else:
            return 2

    def vleft(self, idx):
        """
        Левый сосед вершины с заданным индексом
        """
        return idx - idx % self.sizeX + (idx + self.sizeX - 1) % self.sizeX

    def vright(self, idx):
        """
        Правый сосед вершины с заданным индексом
        """
        return idx - idx % self.sizeX + (idx + 1) % self.sizeX

    def vtop(self, idx):
        """
        Верхний сосед вершины с заданным индексом
        """
        return (idx + - self.sizeX) % self.size

    def vbottom(self, idx):
        """
        Нижний сосед вершины с заданным индексом
        """
        return (idx + self.sizeX) % self.size

    def eleft(self, idx):
        """
        Левое инцидентное ребро для вершины с данным индексом
        :return: глобальный индекс ребра
        """
        return self.size + self.vleft(idx)

    def eright(self, idx):
        """
        Правое инцидентное ребро для вершины с данным индексом
        :return: глобальный индекс ребра
        """
        return self.size + idx

    def etop(self, idx):
        """
        Верхнее инцидентное ребро для вершины с данным индексом
        :return: глобальный индекс ребра
        """
        return self.size * 2 + self.vtop(idx)

    def ebottom(self, idx):
        """
        Нижнее инцидентное ребро для вершины с данным индексом
        :return: глобальный индекс ребра
        """
        return self.size * 2 + idx

    def flefttop(self, idx):
        """
        Левая-верхняя инцидентная ячейка для вершины с данным индексом
        :return: глобальный индекс ячейки
        """
        return self.size * 3 + self.vtop(self.vleft(idx))

    def fleftbottom(self, idx):
        """
        Левая-нижняя инцидентная ячейка для вершины с данным индексом
        :return: глобальный индекс ячейки
        """
        return self.size * 3 + self.vleft(idx)

    def frighttop(self, idx):
        """
        Правая-верхняя инцидентная ячейка для вершины с данным индексом
        :return: глобальный индекс ячейки
        """
        return self.size * 3 + self.vtop(idx)

    def frightbottom(self, idx):
        """
        Правая-нижняя инцидентная ячейка для вершины с данным индексом
        :return: глобальный индекс ячейки
        """
        return self.size * 3 + idx

    def facets(self, idx):
        """
        Гиперграни ячейки с данным индексом
        """
        if self.dim(idx) == 2:
            tmp_idx = idx - 3 * self.size  # индекс верхней левой вершины
            # верхнее, левое, нижнее, правое
            return self.size + tmp_idx, 2 * self.size + tmp_idx, self.size + self.vbottom(
                tmp_idx), 2 * self.size + self.vright(tmp_idx)
        else:
            return self.verts(idx)

    def cofacets(self, idx):
        """
        Пара инцидентных ребру IDX клеток
        :return: список из двух клеток
        """
        if idx < 2 * self.size:  # горизонтальное ребро
            tmp_idx = idx - self.size
            return 3 * self.size + self.vtop(tmp_idx), 3 * self.size + tmp_idx
        else:  # вертикальное ребро
            tmp_idx = idx - 2 * self.size
            return 3 * self.size + self.vleft(tmp_idx), 3 * self.size + tmp_idx

    def verts(self, idx):
        """
        Множество вершин клетки
        """
        if idx < self.size:
            return idx,
        elif idx < 2 * self.size:
            return idx - self.size, self.vright(idx - self.size)
        elif idx < 3 * self.size:
            return idx - 2 * self.size, self.vbottom(idx - 2 * self.size)
        else:
            return idx - 3 * self.size, self.vright(idx - 3 * self.size), \
                   self.vbottom(self.vright(idx - 3 * self.size)), self.vbottom(idx - 3 * self.size)

    def extvalue(self, idx):
        """
        Расширенное значение по глобальному индексу клетки.
        Значения в вершинах клетки по убыванию.
        """
        v = self.verts(idx)
        return tuple(sorted([self.value(v[i]) for i in range(len(v))], reverse=True))

    def coordx(self, idx):
        """
        Координата X вершины
        """
        return idx // self.sizeY

    def coordy(self, idx):
        """
        Координата Y вершины
        """
        return idx % self.sizeX

    def value(self, idx):
        """
        Значение по глобальному индексу вершины.
        """
        return self.values[self.coordx(idx), self.coordy(idx)]

    def is_critical(self, idx):
        """
        Проверяем, является ли клетка с данным индексом критической.
        """
        return self.cr_id[idx]

    def set_critical(self, idx):
        """
        Указываем, что клетка является критической
        """
        if not self.is_critical(idx):
            self.cr_id[idx] = True
            self.cr_cells.append(idx)

    def is_unpaired(self, idx):
        """
        Проверяем, является ли клетка спаренной или помеченной как критическая
        """
        return self.V[idx] < 0 and not self.is_critical(idx)

    def unpaired_facets(self, idx, s):
        """
        Неспаренные гиперграни
        (в статье имеется в виду именно это, а не просто грани)
        данной 2-клетки,
        принадлежащих множеству s
        """
        facets = self.facets(idx)
        return [f for f in facets if f in s and self.is_unpaired(f)]

    def add_gradient_arrow(self, start, end):
        """
        Добавляем стрелку градиента
        """
        self.V[start] = end
        self.V[end] = start

    def star(self, idx):
        """
        Звезда вершины idx
        :return: набор клеток звезды, содержащих данную вершину
        """
        return [self.eright(idx), self.etop(idx), self.eleft(idx), self.ebottom(idx),
                self.frighttop(idx), self.flefttop(idx), self.fleftbottom(idx), self.frightbottom(idx)]

    def lower_star(self, idx):
        """
        Вычисление нижней звезды вершины idx
        Список отсортирован по значению extval, т. е. первый элемент - ребро с наименьшим значением.
        :return:
        """
        v = self.value(idx)
        is_left_lower = self.value(self.vleft(idx)) < v
        is_top_lower = self.value(self.vtop(idx)) < v
        is_right_lower = self.value(self.vright(idx)) < v
        is_bottom_lower = self.value(self.vbottom(idx)) < v
        star = []
        if is_left_lower:
            star.append(self.eleft(idx))
        if is_top_lower:
            star.append(self.etop(idx))
        if is_right_lower:
            star.append(self.eright(idx))
        if is_bottom_lower:
            star.append(self.ebottom(idx))
        if is_left_lower and is_top_lower:
            star.append(self.flefttop(idx))
        if is_right_lower and is_top_lower:
            star.append(self.frighttop(idx))
        if is_left_lower and is_bottom_lower:
            star.append(self.fleftbottom(idx))
        if is_right_lower and is_bottom_lower:
            star.append(self.frightbottom(idx))
        star.sort(key=(lambda x: self.extvalue(x)))
        return star

    def process_lower_stars(self):
        # Две вспомогательные кучи
        pq_zero, pq_one = [], []

        for idx in range(self.size):
            lstar = self.lower_star(idx)
            if len(lstar) == 0:
                self.set_critical(idx)  # Если значение в вершине меньше, чем во всех соседних, то она - минимум.
            else:
                delta = lstar[0]  # Ребро с наименьшим значением
                self.add_gradient_arrow(idx, delta)
                for i in range(1, len(lstar)):
                    if self.dim(lstar[i]) == 1:  # Остальные 1-клетки кладём в pq_zero
                        # Первое значение - ключ для сортировки кучи
                        heapq.heappush(pq_zero, (self.extvalue(lstar[i]), lstar[i]))
                # Ко-грани ребра delta
                cf = self.cofacets(delta)
                for f in cf:
                    if f in lstar and len(self.unpaired_facets(f, lstar)) == 1:
                        heapq.heappush(pq_one, (self.extvalue(f), f))
                while pq_one or pq_zero:
                    while pq_one:
                        alpha = heapq.heappop(pq_one)
                        unpair_facets = self.unpaired_facets(alpha[1], lstar)
                        if not unpair_facets:
                            heapq.heappush(pq_zero, alpha)
                        else:
                            pair = unpair_facets[0]
                            self.add_gradient_arrow(pair, alpha[1])
                            # TODO: remove pair from pq_zero faster
                            pq_zero = [x for x in pq_zero if x[1] != pair]
                            heapq.heapify(pq_zero)
                            for beta in lstar:
                                if len(self.unpaired_facets(beta, lstar)) == 1 and \
                                        (alpha in self.facets(beta) or pair in self.facets(beta)):
                                    heapq.heappush(pq_one, (self.extvalue(beta), beta))
                    if pq_zero:
                        gamma = heapq.heappop(pq_zero)[1]
                        self.set_critical(gamma)
                        for alpha in lstar:
                            if gamma in self.facets(alpha) and self.unpaired_facets(alpha, lstar) == 1:
                                heapq.heappush(pq_one, (self.extvalue(alpha), alpha))

    def print(self):
        print(self.values)

    def draw(self):
        plt.figure()
        plt.pcolor(m.values, cmap="Blues")
        plt.scatter(m.critical_points(0)[1], m.critical_points(0)[0], c='b')
        plt.scatter(m.critical_points(1)[1], m.critical_points(1)[0], c='g')
        plt.scatter(m.critical_points(2)[1], m.critical_points(2)[0], c='r')
        plt.show()

np.set_printoptions(precision=3)
size = 100
sincos = lambda x, y: np.sin(x * 0.1) + np.cos(y * 0.238884)
field = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        field[i, j] = sincos(i, j)
m = TorusMesh(size, size)
m.set_values(field)
m.process_lower_stars()
m.draw()
