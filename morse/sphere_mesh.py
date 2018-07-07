import numpy as np


class SphereMesh:
    """
    Прямоугольная сетка с топологией сферы.
    Добавляется бесконечно удалённая вершина - внешность.
    """

    def __init__(self, lx, ly):
        """
        Создание сетки с нулевыми значениями в клетках.
        Расположение осей:
        0------Y
        |
        |
        |
        X
        Индексация:
        0      1      ...  ly - 1
        ly     ly + 1 ...  2 * ly
        ...    ...    ...  ...
        ...    ...    ...  ...
        Внешняя вершина:
        lx * ly
        Количество вершин: lx * ly + 1, из них одна - внешняя
        Количество рёбер: 2 * lx * ly + lx + ly - 4, из них 2 * (lx + ly - 2) - внешние
        Количество граней: lx * ly + lx + ly - 3, из них 2 * (lx + ly - 2) - внешние
        :param lx: Размер сетки по X
        :param ly: Размер сетки по Y
        :return:
        """
        self.sizeX = lx  # Размеры сетки по X и Y
        self.sizeY = ly
        self.size = lx * ly + 1     # Количество вершин
        self.out_index = lx * ly    # Индекс бесконечно удалённой вершины
        self.edge_num = 2 * lx * ly + lx + ly - 4    # Количество рёбер
        self.face_num = lx * ly + lx + ly - 3        # Количество граней
        self.values = np.zeros((self.sizeX, self.sizeY))    # Значения сетки
        self.cr_cells = []  # Список критических клеток
        self.V = [None] * (4 * lx * ly + 2 * lx + 2 * ly - 6)   # Дискретный градиент
        self.cr_id = np.zeros(4 * lx * ly + 2 * lx + 2 * ly - 6, dtype=bool)   # Индикатор критических клеток
        self.msgraph = None     # Граф Морса-Смейла
        self.ppairs = []        # Список персистентных пар
        self.arcs = {}  # Дуги комплекса Морса-Смейла

    def set_values(self, val):
        """
        :param val: NumPy array
        """
        self.values = val

    def _init_border_vertices(self):
        """
        Список граничных вершин.
        :return:
        """
        self.border_vertices = []
        # Обходим границу по часовой стрелке.
        # Верхние вершины
        for idx in range(self.sizeX - 1):
            self.border_vertices.append(idx)
        # Правые вершины
        for idx in range(1, self.sizeY):
            self.border_vertices.append(idx * self.sizeX - 1)
        # Нижние вершины (справа налево)
        for idx in reversed(range(self.sizeX * (self.sizeY - 1) + 1, self.sizeX * self.sizeY)):
            self.border_vertices.append(idx)
        # Левые вершины (снизу вверх)
        for idx in reversed(range(1, self.sizeY)):
            self.border_vertices.append(idx * self.sizeX)

    def _init_border_edges(self):
        """
        Список граничных рёбер.
        :return:
        """
        pass

    def _init_border_edge_neighbors(self):
        """
        Ставим в соответствие каждой граничной вершине ребро к бесконечно удалённой вершине.
        :return:
        """
        self.border_neighbors = dict()
        edge_counter = self.size + 2 * self.sizeX * self.sizeY - self.sizeX - self.sizeY
        for idx in self.border_vertices:
            self.border_neighbors[idx] = edge_counter
            edge_counter += 1

    def _init_border_facet_neighbors(self):
        """
        Ставим в соответствие каждому граничному ребру грань к бесконечно удалённой вершине.
        :return:
        """
        pass
        # edge_counter = self.size + self.edge_num + (self.sizeX - 1) * (self.sizeY - 1)

    def _is_top_bound(self, idx):
        """
        Проверка, лежит ли вершина на верхней границе
        :param idx:
        :return:
        """
        return idx < self.sizeX

    def _is_bottom_bound(self, idx):
        """
        Проверка, лежит ли вершина на нижней границе
        :param idx:
        :return:
        """
        return idx >= self.sizeX * (self.sizeY - 1)

    def _is_left_bound(self, idx):
        """
        Проверка, лежит ли вершина на левой границе
        :param idx:
        :return:
        """
        return idx % self.sizeX == 0

    def _is_right_bound(self, idx):
        """
        Проверка, лежит ли вершина на правой границе
        :param idx:
        :return:
        """
        return idx % self.sizeX == self.sizeX - 1

    def _vleft(self, idx):
        """
        Левая вершина-сосед (если есть).
        :param idx:
        :return:
        """
        return idx - 1 if not self._is_left_bound(idx) else None

    def _vright(self, idx):
        """
        Правая вершина-сосед (если есть).
        :param idx:
        :return:
        """
        return idx + 1 if not self._is_right_bound(idx) else None

    def _vbottom(self, idx):
        """
        Нижняя вершина-сосед (если есть).
        :param idx:
        :return:
        """
        return idx + self.sizeX if not self._is_bottom_bound(idx) else None

    def _vtop(self, idx):
        """
        Верхняя вершина-сосед (если есть).
        :param idx:
        :return:
        """
        return idx - self.sizeX if not self._is_top_bound(idx) else None

    def _eleft(self, idx):
        """
        Левое ребро-сосед (если есть).
        :param idx:
        :return:
        """
        pass

    def _eright(self, idx):
        """
        Правое ребро-сосед (если есть).
        :param idx:
        :return:
        """
        pass

    def _etop(self, idx):
        """
        Верхнее ребро-сосед (если есть).
        :param idx:
        :return:
        """
        pass

    def _ebottom(self, idx):
        """
        Нижнее ребро-сосед (если есть).
        :param idx:
        :return:
        """
        pass

    def dim(self, idx):
        """
        Размерность клетки
        """
        if idx < self.size:
            return 0
        elif idx < self.size + self.edge_num:
            return 1
        else:
            return 2

    def coordx(self, idx):
        """
        Координата X вершины
        """
        return idx % self.sizeX

    def coordy(self, idx):
        """
        Координата Y вершины
        """
        return idx // self.sizeX

    def value(self, idx):
        """
        Значение по глобальному индексу вершины.
        """
        return self.values[self.coordx(idx), self.coordy(idx)]

    def lower_star(self, idx):
        """
        Вычисление нижней звезды вершины idx
        Список отсортирован по значению extval, т. е. первый элемент - ребро с наименьшим значением.
        :return:
        """
        pass