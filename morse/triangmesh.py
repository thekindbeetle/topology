import matplotlib.pyplot as plt
import matplotlib.collections as mc
import numpy as np
import functools


CONDITIONS = ('torus', 'plain', 'cylinder_x', 'cylinder_y')


class TriangMesh:
    """
    Треугольная сетка.
    Квадратная сетка, разделённая диагоналями.
    """

    def __init__(self, lx, ly, conditions='torus'):
        """
        Создание сетки с нулевыми значениями в клетках.
        Расположение осей:
        0------Y
        |
        |
        |
        X
        Индексация:
        0      1      ...  ly
        ly     ly + 1 ...  2 * ly
        ...    ...    ...  ...
        ...    ...    ...  ...
        :param lx: Размер сетки по X
        :param ly: Размер сетки по Y
        :return:
        """
        self.conditions = conditions  # Граничные условия
        self.sizeX = lx  # Размеры сетки по X и Y
        self.sizeY = ly
        self.size = lx * ly  # Количество вершин
        self.fields = list()  # Список полей, определённых на данной сетке
        if conditions == 'torus':
            self._generate_torus_edges()
        elif conditions == 'plain':
            self._generate_plain_edges()
        elif conditions == 'cylinder_x':
            self._generate_cylinder_x_edges()
        elif conditions == 'cylinder_y':
            self._generate_cylinder_y_edges()
        self.jacobi_set = list()
        print('Mesh of size {0}x{1} created'.format(lx, ly))

    def _generate_torus_edges(self):
        """
        Набор рёбер, соответствующий сетке на торе.
        :return:
        """
        self.hor_edges = [(row * self.sizeY + idx, row * self.sizeY + (idx + 1) % self.sizeY)
                          for idx in range(self.sizeY) for row in range(self.sizeX)]
        self.ver_edges = [(idx, (idx + self.sizeY) % self.size) for idx in range(self.size)]
        self.diag_edges = [(idx,
                            (((idx + 1) % self.sizeY) + ((idx // self.sizeY + 1) % self.size) * self.sizeY) % self.size)
                           for idx in range(self.size)]

    def _generate_plain_edges(self):
        """
        Набор рёбер, соответствующий сетке на прямоугольнике.
        :return:
        """
        self.hor_edges = [(idx, idx + 1) for idx in range(self.size) if (idx + 1) % self.sizeY]
        self.ver_edges = [(idx, idx + self.sizeY) for idx in range(self.size - self.sizeY)]
        self.diag_edges = [(idx, idx + self.sizeY + 1) for idx in range(self.size - self.sizeY - 1)
                           if (idx + 1) % self.sizeY]

    def _generate_cylinder_x_edges(self):
        """
        Набор рёбер, соответствующий сетке на цилиндре (склейка по верхней границе).
        :return:
        """
        self.hor_edges = [(idx, idx + 1) for idx in range(self.size) if (idx + 1) % self.sizeY]
        self.ver_edges = [(idx, (idx + self.sizeY) % self.size) for idx in range(self.size)]
        self.diag_edges = [(idx, (idx + self.sizeY + 1) % self.size) for idx in range(self.size)
                           if (idx + 1) % self.sizeY]

    def _generate_cylinder_y_edges(self):
        """
        Набор рёбер, соответствующий сетке на цилиндре (склейка по боковой границе).
        :return:
        """
        self.hor_edges = [(row * self.sizeY + idx, row * self.sizeY + (idx + 1) % self.sizeY)
                          for idx in range(self.sizeY) for row in range(self.sizeX)]
        self.ver_edges = [(idx, idx + self.sizeY) for idx in range(self.size - self.sizeY)]
        self.diag_edges = [(idx,
                            (((idx + 1) % self.sizeY) + ((idx // self.sizeY + 1) % self.size) * self.sizeY) % self.size)
                           for idx in range(self.size - self.sizeY)]


    def set_field(self, field):
        """
        :param field: NumPy array
        """
        self.fields.append(field)

    @functools.lru_cache(maxsize=None)
    def value(self, field_idx, vert_idx):
        """
        Значение по глобальному индексу вершины для данного поля.
        :param vert_idx: Индекс вершины.
        :param field_idx: Индекс поля.
        """
        return self.fields[field_idx][self.coordy(vert_idx), self.coordx(vert_idx)]

    @functools.lru_cache(maxsize=6)
    def coordx(self, idx):
        """
        Координата X вершины
        """
        return idx % self.sizeY

    @functools.lru_cache(maxsize=6)
    def coordy(self, idx):
        """
        Координата Y вершины
        """
        return idx // self.sizeY

    def edges(self):
        """
        Производящая функция рёбер.
        :return:
        """
        for edge in self.hor_edges:
            yield edge
        for edge in self.ver_edges:
            yield edge
        for edge in self.diag_edges:
            yield edge

    def _coords(self, idx):
        """
        Координаты вершины
        """
        return self.coordx(idx), self.coordy(idx)

    def _vleft(self, idx):
        """
        Левый сосед вершины с заданным индексом
        """
        return idx - idx % self.sizeY + (idx + self.sizeY - 1) % self.sizeY

    def _vright(self, idx):
        """
        Правый сосед вершины с заданным индексом
        """
        return idx - idx % self.sizeY + (idx + 1) % self.sizeY

    def _vtop(self, idx):
        """
        Верхний сосед вершины с заданным индексом
        """
        return (idx + self.size - self.sizeY) % self.size

    def _vbottom(self, idx):
        """
        Нижний сосед вершины с заданным индексом
        """
        return (idx + self.sizeY) % self.size

    def _hor_edgelink(self, edge):
        """
        Линк горизонтального ребра (две вершины для тора).
        Вызывать только для существующих рёбер.
        :param edge: Ребро.
        :return: Пара вершин — линк ребра.
        """
        return self._vtop(edge[0]), self._vbottom(edge[1])

    def _ver_edgelink(self, edge):
        """
        Линк вертикального ребра (две вершины для тора).
        Вызывать только для существующих рёбер.
        :param edge: Ребро.
        :return: Пара вершин — линк ребра.
        """
        return self._vleft(edge[0]), self._vright(edge[1])

    def _diag_edgelink(self, edge):
        """
        Линк диагонального ребра (две вершины для тора).
        Вызывать только для существующих рёбер.
        :param edge: Ребро.
        :return: Пара вершин — линк ребра.
        """
        return self._vright(edge[0]), self._vleft(edge[1])

    def cmp_gradient_at_point(self, field_idx, idx):
        """
        Вектор градиента в точке по его 8-окрестности.
        :param field_idx:
            Индекс поля.
        :param idx:
            Индекс вершины.
        :return:
        """
        # Компонента Y вектора градиента
        # Верхняя и нижняя вершины
        y = self.value(field_idx, self._vtop(idx)) - self.value(field_idx, self._vbottom(idx)) + 0.5 * (
            # Диагональные вершины. Делим на 2, т. к. удалённость на sqrt(2) и при проецировании ещё делим на sqrt(2).
            self.value(field_idx, self._vleft(self._vtop(idx))) + self.value(field_idx, self._vright(self._vtop(idx))) -
            self.value(field_idx, self._vleft(self._vbottom(idx))) - self.value(field_idx, self._vright(self._vbottom(idx)))
        )

        x = self.value(field_idx, self._vright(idx)) - self.value(field_idx, self._vleft(idx)) + 0.5 * (
            # Диагональные вершины. Делим на 2, т. к. удалённость на sqrt(2) и при проецировании ещё делим на sqrt(2).
            self.value(field_idx, self._vbottom(self._vright(idx))) + self.value(field_idx, self._vtop(self._vright(idx))) -
            self.value(field_idx, self._vbottom(self._vleft(idx))) - self.value(field_idx, self._vtop(self._vleft(idx)))
        )

        return x, y

    def cmp_gradient_measure_at_point(self, idx, field_idx1=0, field_idx2=1):
        """
        Вычислить градиентную меру в точке idx для двух полей.
        :param idx:
            Индекс вершины.
        :param field_idx1:
            Индекс первого поля.
        :param field_idx2:
            Индекс второго поля.
        :return:
        """
        x1, y1 = self.cmp_gradient_at_point(field_idx1, idx)
        x2, y2 = self.cmp_gradient_at_point(field_idx2, idx)
        return float(np.cross((x1, y1), (x2, y2)))

    def cmp_gradient_measure(self, field_idx1=0, field_idx2=1):
        """
        Вычислить градиентную меру между двумя полями.
        :param field_idx1: индекс первого поля.
        :param field_idx2: индекс второго поля.
        :return:
        """
        m = np.zeros(self.size)
        for idx in range(self.size):
            x1, y1 = self.cmp_gradient_at_point(field_idx1, idx)
            x2, y2 = self.cmp_gradient_at_point(field_idx2, idx)
            m[idx] = float(np.cross((x1, y1), (x2, y2)))
        return m

    def cmp_jacobi_set(self, field_idx1=0, field_idx2=1, eps=None, log=False):
        """
        Вычисление множества Якоби для сетки на плоскости.
        Подробности алгоритма см. в статье
        Jacobi Sets of Multiple Morse Functions.
        H. Edelsbrunner, J. Harer.
        :param log: Включить текстовый вывод.
        :param eps: Если значения поля на концах ребра отличается менее, чем на eps,
                    они считается равными.
        :param field_idx1: Индекс первого поля.
        :param field_idx2: Индекс второго поля.
        :return:
        """
        result = []
        use_epsilon = eps is not None

        def check_edge(e, link, use_epsilon=False):
            dif_field1 = self.value(field_idx1, e[1]) - self.value(field_idx1, e[0])
            dif_field2 = self.value(field_idx2, e[1]) - self.value(field_idx2, e[0])
            if use_epsilon and np.abs(self.value(field_idx2, e[1]) - self.value(field_idx2, e[0])) < eps:
                # Если функция g равна на концах ребра, то \phi = g
                phi_a = self.value(field_idx2, link[0])
                phi_b = self.value(field_idx2, link[1])
                phi_u = self.value(field_idx2, e[0])
            elif use_epsilon and np.abs(self.value(field_idx1, e[1]) - self.value(field_idx1, e[0])) < eps:
                # Если функция f равна на концах ребра, то \phi = f
                phi_a = self.value(field_idx1, link[0])
                phi_b = self.value(field_idx1, link[1])
                phi_u = self.value(field_idx1, e[0])
            else:
                # Ищем коэффициент l = \lambda такой, что
                # функция \phi = f + \lambda * g принимает одинаковые значения на концах ребра
                l = -dif_field1 / dif_field2

                phi_a = self.value(field_idx1, link[0]) + l * self.value(field_idx2, link[0])
                phi_b = self.value(field_idx1, link[1]) + l * self.value(field_idx2, link[1])
                phi_u = self.value(field_idx1, e[0]) + l * self.value(field_idx2, e[0])

            # если в нижнем линке ребра uv относительно функции \phi
            # ноль или две точки, то ребро принадлежит множеству Якоби
            if not (phi_u - phi_a > 0) ^ (phi_u - phi_b > 0):
                result.append(e)

        print('Horizontal edges...')
        for edge in self.hor_edges:
            check_edge(edge, self._hor_edgelink(edge), use_epsilon=use_epsilon)
        print('Vertical edges...')
        for edge in self.ver_edges:
            check_edge(edge, self._ver_edgelink(edge), use_epsilon=use_epsilon)
        print('Diagonal edges...')
        for edge in self.diag_edges:
            check_edge(edge, self._diag_edgelink(edge), use_epsilon=use_epsilon)
        print('Completed.')
        self.jacobi_set = result

    def _is_edge_internal(self, edge):
        """
        Проверка, пересекает ли ребро границу решётки.
        :param edge: Ребро.
        :return:
        """
        return edge[0] < edge[1] and edge[0] % self.sizeY <= edge[1] % self.sizeY

    def _construct_collection(self, edges):
        """
        Построить LineCollection по набору рёбер.
        :param edges: Список рёбер.
        :return: LineCollection.
        """
        return mc.LineCollection(map(lambda e: tuple(map(self._coords, e)),
                                     [e for e in edges if self._is_edge_internal(e)]), colors='k', linewidths=1)

    def draw(self, field_idx=0, draw_image=True, draw_grid=False, annotate_points=False, fname=None, draw_jacobi_set=(0, 1)):
        plt.style.use('ggplot')
        plt.figure(figsize=(25.1, 35.4), dpi=100)
        ax = plt.gca()
        if draw_image:
            # plt.imshow(self.fields[field_idx])
            plt.pcolor(self.fields[field_idx])
            plt.colorbar()
        plt.xlim((-1, self.sizeY))
        plt.ylim((-1, self.sizeX))
        if draw_grid:
            ax.add_collection(self._construct_collection(list(self.edges())))
        if annotate_points:
            for idx in range(self.size):
                ax.text(*self._coords(idx), str(idx))
        if draw_jacobi_set:
            ax.add_collection(self._construct_collection(self.jacobi_set))
        if fname:
            plt.savefig(fname)
            plt.close()

    @staticmethod
    def build_all(field1, field2, conditions='plain'):
        """
        Создание поля и построение множества Якоби (одновременно).
        :param field:
        :return:
        """
        tr_mesh = TriangMesh(*field1.shape, conditions=conditions)
        tr_mesh.set_field(field1)
        tr_mesh.set_field(field2)
        tr_mesh.cmp_jacobi_set()
        return tr_mesh

# field = np.zeros((3, 4))
# t = TriangMesh(3, 4, conditions='cylinder_y')
# t.set_field(field)
# print(t.hor_edges)
# print(t.ver_edges)
# print(t.diag_edges)
# t.draw(draw_grid=True)
# plt.show()

