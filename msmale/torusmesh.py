import matplotlib.pyplot as plt
import matplotlib.collections as mc
import numpy as np
import heapq
from bitarray import bitarray
from collections import deque
import msmale.unionfind
import copy
import re


class TorusMesh:
    """
    Прямоугольная сетка на торе
    """
    # Размеры сетки по X и Y
    sizeX, sizeY = 0, 0

    # Количество вершин
    size = 0

    # Список критических клеток
    cr_cells = []

    # Дискретный градиент
    V = []

    # Значения сетки
    values = None

    # Индикатор критических клеток
    cr_id = []

    # Список персистентных пар
    ppairs = []

    # Обратное отображение для списка критических точек
    idx_to_cidx = []

    # Граф Морса-Смейла
    msgraph = None

    # Метки критических клеток (негативная / позитивная)
    _signs = None

    # Дуги комплекса Морса-Смейла
    arcs = []

    def __init__(self, lx, ly):
        self.sizeX = lx
        self.sizeY = ly
        self.size = lx * ly
        self.values = np.zeros((self.sizeX, self.sizeY))
        self.cr_cells = []
        self.V = [None] * (4 * self.size)
        self.cr_id = np.zeros(4 * self.size, dtype=bool)
        self.idx_to_cidx = []
        self.msgraph = None
        self.arcs = []

    def set_values(self, val):
        """
        :param val: NumPy array
        """
        self.values = val

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

    def cp(self, morse_index):
        """
        Вывести критические точки с данным индексом Морса.
        :param morse_index: Индекс Морса критической точки
        :return: список индексов критических точек
        """
        return [p for p in self.cr_cells if self.dim(p) == morse_index]

    def to_index(self, x, y):
        """
        Глобальный индекс вершины по координатам сетки.
        """
        return x * self.sizeY + y

    def coords(self, idx):
        """
        Координаты центра клетки (вершины, ребра или ячейки)
        X и Y меняем местами!
        """
        if idx < self.size:
            return self.coordy(idx), self.coordx(idx)
        elif idx < self.size * 2:
            return self.coordy(self.verts(idx)[0]), self.coordx(self.verts(idx)[0]) + 0.5
        elif idx < self.size * 3:
            return self.coordy(self.verts(idx)[0]) + 0.5, self.coordx(self.verts(idx)[0])
        else:
            return self.coordy(self.verts(idx)[0]) + 0.5, self.coordx(self.verts(idx)[0]) + 0.5

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
        return (idx + self.size - self.sizeX) % self.size

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
        @tested
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
        @tested
        Пара инцидентных ребру IDX клеток
        :return: список из двух клеток
        """
        if self.dim(idx) != 1:
            raise AssertionError("Morse index must be 1!")
        if idx < 2 * self.size:  # горизонтальное ребро
            tmp_idx = idx - self.size
            return 3 * self.size + self.vtop(tmp_idx), 3 * self.size + tmp_idx
        else:  # вертикальное ребро
            tmp_idx = idx - 2 * self.size
            return 3 * self.size + self.vleft(tmp_idx), 3 * self.size + tmp_idx

    def verts(self, idx):
        """
        @tested
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

    def is_critical(self, idx):
        """
        Проверяем, является ли клетка с данным индексом критической.
        """
        return self.cr_id[idx]

    def set_critical(self, idx):
        """
        Указываем, что клетка является критической
        """
        self.cr_id[idx] = True
        self.cr_cells.append(idx)

    def is_unpaired(self, idx):
        """
        Проверяем, является ли клетка спаренной или помеченной как критическая
        """
        return (self.V[idx] is None) and (not self.is_critical(idx))

    def unpaired_facets(self, idx, s):
        """
        Неспаренные гиперграни
        (в статье имеется в виду именно это, а не просто грани)
        данной 1- или 2-клетки,
        принадлежащих множеству s
        """
        facets = self.facets(idx)
        return [f for f in facets if f in s and self.is_unpaired(f)]

    def set_gradient_arrow(self, start, end):
        """
        Добавляем стрелку градиента
        """
        self.V[start] = end
        self.V[end] = start

    def get_min_neib_msgraph(self, idx):
        """
        Соседи-минимумы седла в МС-графе.
        Выводятся индексы в списке crit_cells.
        :return:
        """
        if self.dim(self.cr_cells[idx]) != 1:
            raise AssertionError("Функция get_min_neib_msgraph должна вызываться с аргументом-седлом")
        result = [cell for cell in self.msgraph[idx] if self.dim(self.cr_cells[cell]) == 0]
        if len(result) != 2:
            raise ValueError("Ошибка: у седла должны быть ровно два соседа-минимума!")
        return result

    def get_max_neib_msgraph(self, idx):
        """
        Соседи-максимумы седла в МС-графе.
        Выводятся индексы в списке crit_cells.
        :return:
        """
        if self.dim(self.cr_cells[idx]) != 1:
            raise AssertionError("Функция get_min_neib_msgraph должна вызываться с аргументом-седлом")
        result = [cell for cell in self.msgraph[idx] if self.dim(self.cr_cells[cell]) == 2]
        if len(result) != 2:
            raise ValueError("Ошибка: у седла должны быть ровно два соседа-максимума!")
        return result

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
        is_left_bottom_lower = self.value(self.vleft(self.vbottom(idx))) < v
        is_right_bottom_lower = self.value(self.vright(self.vbottom(idx))) < v
        is_left_top_lower = self.value(self.vleft(self.vtop(idx))) < v
        is_right_top_lower = self.value(self.vright(self.vtop(idx))) < v
        star = []
        if is_left_lower:
            star.append(self.eleft(idx))
        if is_top_lower:
            star.append(self.etop(idx))
        if is_right_lower:
            star.append(self.eright(idx))
        if is_bottom_lower:
            star.append(self.ebottom(idx))
        if is_left_lower and is_top_lower and is_left_top_lower:
            star.append(self.flefttop(idx))
        if is_right_lower and is_top_lower and is_right_top_lower:
            star.append(self.frighttop(idx))
        if is_left_lower and is_bottom_lower and is_left_bottom_lower:
            star.append(self.fleftbottom(idx))
        if is_right_lower and is_bottom_lower and is_right_bottom_lower:
            star.append(self.frightbottom(idx))
        star.sort(key=(lambda x: self.extvalue(x)))
        return star

    def cmp_discrete_gradient(self):
        """
        Реализация процедуры вычисления дискретного градиента по исходным данным.
        Сюда включено вычисление критических клеток.
        ProcessLowerStars
        (Vanessa Robins)
        :return:
        """
        self.cr_cells = []
        self.V = [None] * (4 * self.size)
        self.cr_id = np.zeros(4 * self.size, dtype=bool)
        self.idx_to_cidx = None

        # Две вспомогательные кучи
        pq_zero = []
        pq_one = []
        percentage = 0

        for idx in range(self.size):
            if idx / self.size > (percentage + 1) * 0.01:
                percentage += 1
                print("Gradient computation... {0}% completed".format(percentage))
            lstar = self.lower_star(idx)
            if not lstar:
                self.set_critical(idx)  # Если значение в вершине меньше, чем во всех соседних, то она - минимум.
            else:
                delta = lstar[0]  # Ребро с наименьшим значением
                self.set_gradient_arrow(idx, delta)
                for i in range(1, len(lstar)):
                    if self.dim(lstar[i]) == 1:  # Остальные 1-клетки кладём в pq_zero
                        # Первое значение - ключ для сортировки кучи
                        heapq.heappush(pq_zero, (self.extvalue(lstar[i]), lstar[i]))
                # Ко-грани ребра delta
                cf = self.cofacets(delta)
                for f in cf:
                    if (f in lstar) and (len(self.unpaired_facets(f, lstar)) == 1):
                        heapq.heappush(pq_one, (self.extvalue(f), f))
                while pq_one or pq_zero:
                    while pq_one:
                        alpha = heapq.heappop(pq_one)
                        unpair_facets = self.unpaired_facets(alpha[1], lstar)
                        if not unpair_facets:
                            heapq.heappush(pq_zero, alpha)
                        else:
                            pair = unpair_facets[0]
                            self.set_gradient_arrow(pair, alpha[1])
                            # TODO: remove pair from pq_zero faster
                            pq_zero = [x for x in pq_zero if x[1] != pair]
                            # pq_zero.remove((self.extvalue(pair), pair))
                            heapq.heapify(pq_zero)
                            for beta in lstar:
                                if len(self.unpaired_facets(beta, lstar)) == 1 and ((alpha in self.facets(beta)) or (pair in self.facets(beta))):
                                    heapq.heappush(pq_one, (self.extvalue(beta), beta))
                    if pq_zero:
                        gamma = heapq.heappop(pq_zero)
                        self.set_critical(gamma[1])
                        for a in lstar:
                            if (gamma[1] in self.facets(a)) and (len(self.unpaired_facets(a, lstar)) == 1):
                                heapq.heappush(pq_one, (self.extvalue(a), a))

        # Сортируем клетки по возрастанию значения — получаем фильтрацию.
        self.cr_cells.sort(key=lambda idx: self.extvalue(idx))

        # Строим обратное отображение для списка критических точек
        self.idx_to_cidx = {self.cr_cells[cidx]: cidx for cidx in range(len(self.cr_cells))}

    def cmp_crit_cells(self):
        """
        Вычисление критических клеток по градиенту.
        ! Функция cmp_discrete_gradient вычисляет критические клетки по данным.
        ! Здесь критические точки вычисляются не по данным, а по градиенту.
        Если клетка не включена ни в одну стрелку градиента, то она критическая.
        Проверка на корректность градиента не производится.
        :return:
        """
        self.cr_cells = []
        self.cr_id = np.ones(4 * self.size, dtype=bool)
        for i in range(4 * self.size):
            if self.V[i] is not None:
                self.cr_id[i] = False
        self.cr_cells = [idx for idx in range(4 * self.size) if self.cr_id[idx]]

        # Сортируем клетки по возрастанию значения — получаем фильтрацию.
        self.cr_cells.sort(key=lambda idx: self.extvalue(idx))

        # Строим обратное отображение для списка критических точек
        self.idx_to_cidx = {self.cr_cells[cidx]: cidx for cidx in range(len(self.cr_cells))}

    def cmp_msgraph(self):
        """
        Вытаскиваем информацию о соседстве в MS-комплексе.
        На выходе - список, каждой критической клетке сопоставляется список её соседей в MS-графе.
        """
        self.msgraph = [[] for i in range(len(self.cr_cells))]

        q = deque()

        for dimension in (1, 2):
            for idx in self.cp(dimension):
                cidx = self.idx_to_cidx[idx]  # Индекс в списке критических точек
                g = self.facets(idx)
                for face in g:
                    if self.is_critical(face):
                        self.msgraph[cidx].append(self.idx_to_cidx[face])
                        self.msgraph[self.idx_to_cidx[face]].append(cidx)
                    elif self.V[face] > face:  # То есть есть стрелка, и она выходит (а не входит)
                        q.appendleft(face)
                while q:
                    a = q.pop()
                    b = self.V[a]
                    for face in self.facets(b):
                        if face == a:
                            continue  # Возвращаться нельзя
                        if self.is_critical(face):
                            self.msgraph[cidx].append(self.idx_to_cidx[face])
                            self.msgraph[self.idx_to_cidx[face]].append(cidx)
                        elif self.V[face] > face:
                            q.appendleft(face)

    def cmp_arcs(self):
        """
        Вычисляем сепаратрисы MS-комплекса.
        Граф к моменту вызова функции должен быть вычислен.
        :return:
        """
        self.arcs = []
        for s in self.cp(1): # Цикл по сёдлам
            # Вычисляем сепаратрисы седло-минимум
            vertices = self.verts(s)

            for cur_v in vertices:
                # Идём в двух возможных направлениях
                cur_e = s
                separx = [cur_e, cur_v]  # Новая сепаратриса
                # Идём по сепаратрисе, пока не встретим критическую клетку

                while not self.is_critical(cur_v):
                    # Идём вдоль интегральной линии
                    cur_e = self.V[cur_v]
                    v = self.verts(cur_e)
                    # Выбираем путь вперёд (в ещё не пройденную клетку)
                    cur_v = v[1] if v[0] == cur_v else v[0]
                    separx.append(cur_e)
                    separx.append(cur_v)
                self.arcs.append(separx)

            # Вычисляем сепаратрисы седло-максимум
            faces = self.cofacets(s)

            # Идём в двух возможных направлениях
            for cur_f in faces:
                cur_e = s
                separx = [cur_e, cur_f] # Новая сепаратриса
                while not self.is_critical(cur_f):
                    # Идём вдоль интегральной линии
                    cur_e = self.V[cur_f]
                    f = self.cofacets(cur_e)
                    # Выбираем путь вперёд (в ещё не пройденную клетку)
                    cur_f = f[1] if f[0] == cur_f else f[0]
                    separx.append(cur_e)
                    separx.append(cur_f)
                self.arcs.append(separx)

    def assign_labels(self):
        """
        Пометить критические клетки как негативные (создающие цикл) или позитивные (убивающие цикл).
        :return:
        """
        # Инициализация массива отметок
        self._signs = bitarray(len(self.cr_cells))

        uf = msmale.unionfind.UnionFind(len(self.cr_cells))

        # Cчитаем, что к этому моменту массив критических точек представляет собой фильтрацию.
        for i in range(len(self.cr_cells)):
            cc = self.cr_cells[i]
            dim = self.dim(cc)  # размерность критической клетки
            uf.makeset(i)
            if dim == 0:
                self._signs[i] = 1 # все минимумы - позитивные
            elif dim == 1:
                # находим двух соседей в ms-комплексе, которые являются минимумами
                # у каждого седла два соседа-минимума и два соседа-максимума
                neighbours = self.get_min_neib_msgraph(i)  # соседи-минимумы в графе ms-комплекса
                if uf.find(neighbours[0]) == uf.find(neighbours[1]):
                    self._signs[i] = 1  # седло порождает 1-цикл
                else:
                    self._signs[i] = 0  # седло убивает 0-цикл
                uf.union(neighbours[0], neighbours[1])  # Объединяем компоненты связности
            else:
                self._signs[i] = 0  # все максимумы - негативные

    def cmp_persistent_pairs(self):
        """
        Вычисление персистентных пар
        :return:
        """
        # Несортированный массив персистентных пар
        pairs = []

        # список циклов
        # каждый цикл соответствует негативной клетке
        cycles = [None] * len(self.cr_cells)

        curset = len(self.cr_cells) * bitarray('0')

        # Последнее вхождение единицы в битовый массив (если её нет — ValueError)
        get_highest_of = lambda bitset: bitset.to01().rindex('1')

        # Первое вхождение единицы в битовый массив (если её нет — ValueError)
        get_lowest_of = lambda bitset: bitset.to01().index('1')

        # Персистентность пары
        pers = lambda idx1, idx2: np.abs(np.mean(self.extvalue(self.cr_cells[idx1])) - np.mean(self.extvalue(self.cr_cells[idx2])))

        # проходим по прямой фильтрации
        for i in range(len(self.cr_cells)):
            cc = self.cr_cells[i]
            # Смотрим только негативные сёдла
            if self.dim(cc) == 1 and not self._signs[i]:
                for neib in self.get_min_neib_msgraph(i):
                    curset[neib] = True   # 5:
                while curset.count() > 0:
                    s = get_highest_of(curset)  # 9:
                    if not cycles[s]:
                        cycles[s] = copy.deepcopy(curset)
                        cycles[i] = copy.deepcopy(curset)
                        pairs.append((i, s, pers(i, s)))
                    else:
                        for b in re.finditer('1', cycles[s].to01()):  # 16:
                            idx = b.span()[0]
                            curset[idx] ^= 1  # bit flip operation

        curset = len(self.cr_cells) * bitarray('0')

        # проходим по обратной фильтрации
        for i in reversed(range(len(self.cr_cells))):
            cc = self.cr_cells[i]
            # Смотрим только позитивные сёдла
            if self.dim(cc) == 1 and self._signs[i]:
                for neib in self.get_max_neib_msgraph(i):
                    curset[neib] = True
                while curset.count() > 0:
                    s = get_lowest_of(curset)
                    if not cycles[s]:
                        cycles[s] = copy.deepcopy(curset)
                        cycles[i] = copy.deepcopy(curset)
                        pairs.append((i, s, pers(i, s)))
                    else:
                        for b in re.finditer('1', cycles[s].to01()):  # 16:
                            idx = b.span()[0]
                            curset[idx] ^= 1  # bit flip operation

        pairs.sort(key=lambda x: x[2], reverse=True)  # Сортируем пары по убыванию значения
        self.ppairs = pairs

    def eliminate_pair_revert_gradient(self):
        """
        Сократить следующую по очерёдности персистентную пару.
        !При помощи разворота градиента!
        На дуге разворачивается градиент.
        :return:
        """
        # Если пар не осталось, то скоращать нечего.
        if not self.ppairs:
            print("Список персистентных пар пуст!")
            return
        # Первая клетка — седло, вторая — максимум или минимум.
        pair = self.ppairs.pop()
        saddle = self.cr_cells[pair[0]]
        if self.dim(saddle) != 1:
            raise AssertionError("Первая клетка пары должна быть седлом!")
        extr = self.cr_cells[pair[1]]
        extr_dim = self.dim(extr)
        # Разворот градиента.
        if extr_dim == 1:
            raise AssertionError("Вторая клетка пары должна быть максимумом или минимумом!")
        else:
            # Ищем соответствующую дугу
            curr_arcs = [arc for arc in self.arcs if arc[0] == saddle and arc[-1] == extr]
            if len(curr_arcs) == 0:
                raise RuntimeError("Дуга, соответствующая персистентной паре, не найдена!")
            elif len(curr_arcs) >= 2:
                raise RuntimeError("Дуга, соответствующая персистентной паре, должна быть ровно одна!")
            arc = curr_arcs[0]
            # Разворачиваем градиент вдоль дуги.
            for i in range(0, len(arc), 2):
                self.set_gradient_arrow(arc[i], arc[i + 1])


    def eliminate_pair_change_msgraph(self):
        """
        Сократить следующую по очерёдности персистентную пару.
        !Изменяется граф Морса-Смейла!
        :return:
        """
        # Если пар не осталось, то скоращать нечего.
        if not self.ppairs:
            print("Список персистентных пар пуст!")
            return
        # Первая клетка — седло, вторая — максимум или минимум.
        pair = self.ppairs.pop()
        saddle = pair[0]
        if self.dim(self.cr_cells[saddle]) != 1:
            raise AssertionError("Первая клетка пары должна быть седлом!")
        extr = pair[1]
        extr_dim = self.dim(self.cr_cells[extr])

        # Изменение графа Морса-Смейла.
        if extr_dim == 1:
            raise AssertionError("Вторая клетка пары должна быть максимумом или минимумом!")
        elif extr_dim == 0:
            # Вторая клетка — минимум
            saddles = [x for x in self.msgraph[extr] if x != saddle]  # Сёдла-соседи максимума (кроме седла из пары)
            mins = self.get_min_neib_msgraph(saddle)
            minimum = mins[0] if mins[0] != extr else mins[1]
            for x in self.msgraph[saddle]:
                self.msgraph[x].remove(saddle)
            self.msgraph[saddle].clear()
            for s in saddles:
                # Добавляем рёбра из соседей максимума в другой максимум
                self.msgraph[s].remove(extr)
                self.msgraph[extr].append(s)
                self.msgraph[s].append(minimum)
                self.msgraph[minimum].append(s)
            self.msgraph[extr].clear()
        elif extr_dim == 2:
            saddles = [x for x in self.msgraph[extr] if x != saddle]  # Сёдла-соседи максимума (кроме седла из пары)
            maxs = self.get_max_neib_msgraph(saddle)
            maximum = maxs[0] if maxs[0] != extr else maxs[1]
            for x in self.msgraph[saddle]:
                self.msgraph[x].remove(saddle)
            self.msgraph[saddle].clear()
            for s in saddles:
                # Добавляем рёбра из соседей минимума в другой минимум
                self.msgraph[s].remove(extr)
                self.msgraph[extr].append(s)
                self.msgraph[s].append(maximum)
                self.msgraph[maximum].append(s)
            self.msgraph[extr].clear()
        print("Pair {0} eliminated".format(pair))

    def print(self):
        print(self.values)

    def draw(self, draw_crit_pts=True, annotate_crit_points=False, draw_persistence_pairs=False, draw_gradient=True, draw_arcs=True, draw_graph=False):
        plt.figure()
        cur_plot = plt.pcolor(self.values, cmap="Blues")
        plt.colorbar(cur_plot)
        if draw_graph:
            edges = []
            for cidx in range(len(self.cr_cells)):
                for cidx2 in self.msgraph[cidx]:
                    edges.append([self.coords(self.cr_cells[cidx]), self.coords(self.cr_cells[cidx2])])
            lc = mc.LineCollection(edges, colors='k', linewidths=2, zorder=1)
            plt.gca().add_collection(lc)
        if draw_gradient:
            x, y, X, Y = [], [], [], []
            for idx in range(len(self.V)):
                if self.V[idx] is None:
                    continue
                if idx < self.V[idx]:
                    start = self.coords(idx)
                    end = self.coords(self.V[idx])
                    if start[0] != 0 and end[0] != 0 and start[1] != 0 and end[1] != 0:
                        x.append(start[0])
                        y.append(start[1])
                        X.append(end[0] - start[0])
                        Y.append(end[1] - start[1])
            plt.quiver(x, y, X, Y, scale_units='x', angles='xy', scale=2)
        if draw_arcs:
            edges = []
            for arc in self.arcs:
                for idx in range(len(arc) - 1):
                    edge = [self.coords(arc[idx]), self.coords(arc[idx + 1])]
                    if np.abs(edge[0][0] - edge[1][0]) < 1 and np.abs(edge[0][1] - edge[1][1]) < 1:
                        edges.append([self.coords(arc[idx]), self.coords(arc[idx + 1])])
            lc = mc.LineCollection(edges, colors='k', linewidths=1, zorder=1)
            plt.gca().add_collection(lc)
        if draw_crit_pts:
            plt.scatter([self.coords(p)[0] for p in self.cp(0)], [self.coords(p)[1] for p in self.cp(0)], zorder=2, c='b', s=50)
            plt.scatter([self.coords(p)[0] for p in self.cp(1)], [self.coords(p)[1] for p in self.cp(1)], zorder=2, c='g', s=50)
            plt.scatter([self.coords(p)[0] for p in self.cp(2)], [self.coords(p)[1] for p in self.cp(2)], zorder=2, c='r', s=50)
        if annotate_crit_points:
            for cidx in range(len(self.cr_cells)):
                p = self.cr_cells[cidx]
                plt.text(self.coords(p)[0], self.coords(p)[1], str(cidx))
        if draw_persistence_pairs:
            edges = []
            for pairs in self.ppairs:
                edges.append([self.coords(self.cr_cells[pairs[0]]), self.coords(self.cr_cells[pairs[1]])])
            lc = mc.LineCollection(edges, colors='r', linewidths=2, zorder=1)
            plt.gca().add_collection(lc)
        plt.show()

def test():
    """
    Максимум 27
    два минимума 2, 7
    три седла 15, 16, 23
    """
    field = np.asarray([[2, 8, 1], [9, 5, 6], [7, 3, 4]])
    m = TorusMesh(3, 3)
    m.set_values(field)
    m.cmp_discrete_gradient()
    m.cmp_msgraph()
    m.assign_labels()
    m.cmp_arcs()
    print(m._signs)
    m.draw(draw_gradient=False, draw_arcs=True, draw_graph=False)

#
# import generators.matern
# import msmale.field_generator
#
# centers = generators.matern.matern_point_process(0.5, 10, 3, 20)
# data = msmale.field_generator.gen_gaussian_sum_torus(20, 20, centers, 4)
# msmale.field_generator.perturb(data)
# m = TorusMesh(data.shape[0], data.shape[1])
# m.set_values(data)
# m.print()
# m.cmp_discrete_gradient()
# m.cmp_arcs()
# m.cmp_ms_graph()
# m.assign_labels()
# m.cmp_persistent_pairs()
# print(m.ppairs)
# plt.ion()
# m.draw(draw_arcs=False, draw_gradient=True, draw_crit_pts=True, annotate_crit_points=True, draw_persistence_pairs=True, draw_graph=False)
# for i in range(5):
#     m.eliminate_pair_revert_gradient()
#     m.draw(draw_arcs=False, draw_gradient=True, draw_crit_pts=True, annotate_crit_points=True, draw_persistence_pairs=True, draw_graph=False)
# plt.waitforbuttonpress()