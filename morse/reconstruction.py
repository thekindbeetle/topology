import matplotlib.pyplot as plt
import numpy as np
from morse.torusmesh import TorusMesh


class LaplaceSmoother:
    """
    Сглаживаем методом конечных разностей картинку с заданной маской.
    Значения маски фиксированы. Остальные на каждом шаге определяются как среднее между значениями в окрестности.
    """
    def __init__(self, lx, ly):
        """
        Создать сглаживатель с указанием длины и ширины поля.
        :param lx:  Длина по оси X
        :param ly:  Длина по оси Y
        :return:
        """
        self.lx = lx
        self.ly = ly
        self.field = np.zeros((lx, ly), dtype=float)
        self.mask = np.zeros((lx, ly), dtype=bool)

    def draw(self):
        cur_plot = plt.imshow(self.field, cmap='gray', origin='lower')
        plt.colorbar(cur_plot)

    def set_val(self, x, y, value):
        """
        Устанавливаем значение ячейки (x, y).
        """
        self.field[x, y] = value

    def set_mask(self, x, y):
        """
        Делаем значение ячейки неизменным при переходе между состояниями.
        """
        self.mask[x, y] = True

    def next(self):
        """
        Проводим шаг сглаживания.
        """
        # Складываем поле с самим собой, сдвинутым на 1 во всех направлениях.
        # Делим на 4
        # Получаем то же самое, что свёртка с фильтром Лапласа для 4-соседской структуры на торе.
        # Восстанавливаем значения маскированных ячеек.

        new_field = np.zeros(self.field.shape)
        new_field[:self.lx-1, :self.ly] += self.field[1:self.lx, :self.ly]
        new_field[self.lx-1, :self.ly] += self.field[0, :self.ly]

        new_field[:self.lx, :self.ly-1] += self.field[:self.lx, 1:self.ly]
        new_field[:self.lx, self.ly-1] += self.field[:self.lx, 0]

        new_field[1:self.lx, :self.ly] += self.field[:self.lx-1, :self.ly]
        new_field[0, :self.ly] += self.field[self.lx-1, :self.ly]

        new_field[:self.lx, 1:self.ly] += self.field[:self.lx, :self.ly-1]
        new_field[:self.lx, 0] += self.field[:self.lx, self.ly-1]

        new_field *= 0.25

        # Значение фиксированных клеток не изменяется
        self.field = np.where(self.mask, self.field, new_field)

    @staticmethod
    def create_from_mesh(mesh, lx, ly):
        """
        :type mesh: TorusMesh
        """
        arc_list = mesh.list_arcs()
        arc_list.sort(key=len)
        arc_values_list = []

        # Линейным интерполированием устанавливаем значения на дугах
        for arc in arc_list:
            start_val = mesh._extvalue(arc[0])[0]
            end_val = mesh._extvalue(arc[-1])[0]
            values = np.linspace(start_val, end_val, len(arc))
            arc_values_list.append(values)

        # Создаём экземпляр сглаживателя
        a = LaplaceSmoother(lx, ly)
        for arc, arc_val in zip(arc_list, arc_values_list):
            for idx in range(len(arc)):
                x, y = int(mesh._coords(arc[idx])[1]), int(mesh._coords(arc[idx])[0])
                a.set_val(x, y, arc_val[idx])
                a.set_mask(x, y)
        return a
