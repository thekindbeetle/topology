import geom.vert
import geom.persistence.filtration


class Persistence:

    # Фильтрация комплекса
    filtration = None

    # Cycles at all levels.
    numOfCycles = None

    # Components at all levels.
    numOfComponents = None

    # Big components at all levels.
    numOfBigComponents = None

    # Здесь хранятся индексы компонент связности
    _parentOfSimplex = None

    # Времена рождения компонент связности.
    # Компонента с номером i — это компонента, в которой точка с индексом i является главной.
    # Время рождения i-й компоненты лежит в i-й ячейке списка.
    # Остальные ячейки пусты
    _compBirthTimes = None

    # Времена смерти компонент связности.
    # Компонента с номером i — это компонента, в которой точка с индексом i является главной.
    # Время смерти i-й компоненты лежит в i-й ячейке списка.
    # Остальные ячейки пусты
    _compDeathTimes = None

    # Времена рождения 1-циклов.
    # Цикл с номером i — это цикл, в которой треугольник с индексом i является главным.
    # Время рождения i-го цикла лежит в i-й ячейке списка.
    # Остальные ячейки пусты
    _cycleBirthTimes = None

    # Времена смерти 1-циклов.
    # Цикл с номером i — это цикл, в которой треугольник с индексом i является главным.
    # Время смерти i-го цикла лежит в i-й ячейке списка.
    # Остальные ячейки пусты
    _cycleDeathTimes = None

    def __init__(self, filtration):
        self.filtration = filtration
        simp_num = filtration.simplexes_num()

        self.numOfComponents = [0 for i in range(simp_num)]
        self.numOfBigComponents = [0 for i in range(simp_num)]
        self.numOfCycles = [0 for i in range(simp_num)]
        self._parentOfSimplex = []
        self._compBirthTimes = []
        self._compDeathTimes = []

        self.init_cycles_data()     # инициализация данных о циклах
        self.init_components_data() # инициализация данных о компонентах связности

    def get_diagram(self):
        """
        Диаграмма персистентности.
        :return: набор времён рождения, набор времён смерти.
        """
        x, y = [], []
        for idx in range(len(self._cycleBirthTimes)):
            if self._cycleBirthTimes[idx] != -1 and self._cycleDeathTimes[idx] != -1:
                x.append(self._cycleBirthTimes[idx])
                y.append(self._cycleDeathTimes[idx])
        return x, y

    def _union(self, i, j):
        """
        Слияние компонент связности, содержащих треугольники с номерами i и j в
        фильтрации.
        Умирает тот, кто позже родился, т. е. с меньшим индексом
        (т. к. проходим фильтрацию с конца)
        :param i: номер треугольника в фильтрации
        :param j: номер треугольника в фильтрации
        :return:
        """
        parent_i = self.find(i)
        parent_j = self.find(j)
        min_parent = min(parent_i, parent_j)
        max_parent = max(parent_i, parent_j)
        self._parentOfSimplex[min_parent] = max_parent
        # for k in range(len(self._parentOfSimplex)):
        #     if self._parentOfSimplex[k] == min_parent:
        #         self._parentOfSimplex[k] = max_parent

    def find(self, i):
        """
        Рекурсивный поиск компоненты связности, которой принадлежит треугольник с номером i в фильтрации
        :param i: номер треугольника в фильтрации
        :return:
        """
        curr_idx = i
        while self._parentOfSimplex[curr_idx] != curr_idx:
            curr_idx = self._parentOfSimplex[curr_idx]
        return curr_idx

    def init_cycles_data(self):
        # Инициализируются 2 массива: со временами рождения и временем смерти.
        # В i-той ячейке массива _birthTimes время рождения i-ого цикла,
        # в i-ой ячейке масссива _deathTimes время смерти i-ого цикла.
        # Номер i - порядковый номер цикла, циклы упорядочены по времения смерти от большего к меньшему.
        # Это связано со способом инициализации: проходим фильтрацию в обратном порядке,
        # каждому треугольнику ставим в соответствие цикл.
        # Время появления треугольника - момент смерти соответствующего цикла.
        # Момент смерти цикла определяется как время появления замыкающего ребра.
        self._cycleBirthTimes = [-1 for i in range(self.filtration.simplexes_num())]
        self._cycleDeathTimes = [-1 for i in range(self.filtration.simplexes_num())]

        # Число циклов на текущем уровне (-1, т.к. не считаем цикл, соответствующий внешности)
        curr_cycles_num = -1

        # инициализируем массив родителей
        self._parentOfSimplex = [-1 for i in range(self.filtration.simplexes_num())]

        # Пробегаем фильтрацию симплексов.
        for filt_idx in reversed(range(self.filtration.simplexes_num() - 1)):
            dim = self.filtration.get_simplex(filt_idx).dim
            # Cмотрим на размерность симплексов
            if dim == 2:
                # Если текущий симлекс - треугольник, значит, родилась компонента связности, умер цикл.
                self._parentOfSimplex[filt_idx] = filt_idx
                self._cycleDeathTimes[filt_idx] = self.filtration.get_simplex(filt_idx).appTime
                curr_cycles_num += 1 # Число циклов уменьшилось на 1
            elif dim == 1:
                # Если текущий симлекс - ребро, возможны 2 варианта:
                # оно граничит с двумя компонентами связности или с одной.
                # Индексы треугольников, граничащих по данному ребру
                # (один из них, возможно, — внешность):
                tr_filt_idx_0 = self.filtration.get_inc_triang_of_edge(filt_idx)[0]
                tr_filt_idx_1 = self.filtration.get_inc_triang_of_edge(filt_idx)[1]

                # Проверяем, граничит ли ребро с разными компонентами связности.
                if self.find(tr_filt_idx_0) != self.find(tr_filt_idx_1):
                    # Если да, компоненты сливаются, старшая компонента поглощает младшую.
                    # Значит, родился цикл, соответствующий младшей компоненте связности.
                    self._cycleBirthTimes[min(self.find(tr_filt_idx_0), self.find(tr_filt_idx_1))] =\
                            self.filtration.get_simplex(filt_idx).appTime
                    # Объединение компонент
                    self._union(tr_filt_idx_0, tr_filt_idx_1)
                    curr_cycles_num -= 1  # число циклов увеличилось на 1
            self.numOfCycles[filt_idx - 1] = curr_cycles_num     # количество циклов на текущем уровне

        # Этот кусок кода обрабатываем списки _cycleBirthTimes и _cycleDeathTimes, выкидывая избыточные значения,
        # т. е. те, где время рождения или смерти равно -1.

        # for idx in range(self.filtration.simplexes_num() - 1):  # -1, так как не учитываем внешность
        #     if self._cycleBirthTimes[idx] != -1 and self._cycleDeathTimes[idx] != -1 and\
        #                     self._cycleBirthTimes[idx] != self._cycleDeathTimes[idx]:
        #         self.cycleBirthTimes.add(_cycleBirthTimes[i]);
        #         cycleDeathTimes.add(_cycleDeathTimes[i]);

    def init_components_data(self):
        self._compBirthTimes = [-1 for i in range(self.filtration.simplexes_num())]
        self._compDeathTimes = [-1 for i in range(self.filtration.simplexes_num())]

        # Число компонент связности на текущем уровне
        curr_comp_num = 0

        # Число точек в каждой компоненте связности
        points_count_in_component = [0 for i in range(self.filtration.simplexes_num())]

        # Число больших компонент - компонент связности на текущем уровне, содержащих по крайней мере 3 точки
        curr_big_comp_num = 0
        self._parentOfSimplex = [-1 for i in range(self.filtration.simplexes_num())]

        # Пробегаем фильтрацию симплексов.
        for filt_idx in range(self.filtration.simplexes_num()):
            # Cмотрим на размерность симплексов
            dim = self.filtration.get_simplex(filt_idx).dim
            if dim == 0:
                # Если текущий симлекс - вершина, значит, родилась компонента связности.
                self._parentOfSimplex[filt_idx] = filt_idx
                self._compBirthTimes[filt_idx] = self.filtration.get_simplex(filt_idx).appTime
                points_count_in_component[filt_idx] = 1
                curr_comp_num += 1  # Число компонент связности увеличилось на 1
            elif dim == 1:
                # Если текущий симлекс - ребро, возможны 2 варианта:
                # оно граничит с двумя компонентами связности или с одной.
                # Глобальные индексы вершин данного ребра:
                glob_v_idx_0 = self.filtration.get_simplex(filt_idx).v(0)
                glob_v_idx_1 = self.filtration.get_simplex(filt_idx).v(1)

                # По ним ищем идексы вершин в фильтрации
                filt_v_idx_0 = self.filtration.vertices[glob_v_idx_0].filtInd
                filt_v_idx_1 = self.filtration.vertices[glob_v_idx_1].filtInd

                # Проверяем, граничит ли ребро с разными компонентами связности
                if self.find(filt_v_idx_0) != self.find(filt_v_idx_1):
                    # Если да, компоненты сливаются, старшая компонента поглощает младшую.
                    max_idx = max(self.find(filt_v_idx_0), self.find(filt_v_idx_1)) # Номер старшей компоненты связности
                    min_idx = min(self.find(filt_v_idx_0), self.find(filt_v_idx_1)) # Номер младшей компоненты связности

                    # Если сливаются две компоненты связности, каждая из которых содержит более двух точек,
                    # число больших компонент уменьшается на 1.
                    if points_count_in_component[max_idx] > 2 and points_count_in_component[min_idx] > 2:
                        curr_big_comp_num -= 1
                    # Если сливаются 2 компоненты связности, содержащие по 2 точки или две и одну,
                    # число больших компонент увиличивается на 1.
                    elif points_count_in_component[max_idx] == 2 and points_count_in_component[min_idx] == 2 or\
                            points_count_in_component[max_idx] == 1 and points_count_in_component[min_idx] == 2 or\
                            points_count_in_component[max_idx] == 2 and points_count_in_component[min_idx] == 1:
                        curr_big_comp_num += 1

                    # Объединяем компоненты
                    self._union(filt_v_idx_0, filt_v_idx_1)

                    # При слиянии компонент старшая поглощает младшую.
                    # К количеству точек в старшей добавляется количество точек младшей компоненты.
                    points_count_in_component[max_idx] += points_count_in_component[min_idx]

                    # После этого количество точек младшей компоненты обнуляется.
                    points_count_in_component[min_idx] = 0
                    self._compDeathTimes[min_idx] = self.filtration.get_simplex(filt_idx).appTime
                    curr_comp_num -= 1  # Число компонент связности уменьшилось на 1
            self.numOfComponents[filt_idx] = curr_comp_num
            self.numOfBigComponents[filt_idx] = curr_big_comp_num

        # Этот кусок кода обрабатываем списки _compBirthTimes и _compDeathTimes, выкидывая избыточные значения,
        # т. е. те, где время рождения или смерти равно -1.

        # for (int i = 0; i < _f.simpNumber(); i++) {
        #     if (_compBirthTimes[i] != -1 && _compDeathTimes[i] != -1 && _compBirthTimes[i] != _compDeathTimes[i]) {
        #         compBirthTimes.add(_compBirthTimes[i]);
        #         compDeathTimes.add(_compDeathTimes[i]);
        #     }
        # }


def test():
    import matplotlib.pyplot as plt
    import numpy as np

    points = [
        [1.0, 1.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [5.0, 7.0],
        [9.0, 11.0],
        [10.0, 11.0],
        [-1.0, 8.0]
    ]
    plt.plot(*np.transpose(points), 'ok')
    f = geom.persistence.filtration.Filtration(points)
    f.print()
    pers = Persistence(f)
    print(pers._compBirthTimes)
    print(pers._compDeathTimes)
