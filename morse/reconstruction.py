import matplotlib.pyplot as plt
import numpy as np
from morse.torusmesh import TorusMesh
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings


def test():
    import morse.field_generator as gen
    image = gen.gen_field_from_file(r"C:\repo\pproc\data\input.fits", conditions="plain", filetype="fits")
    mesh = TorusMesh.build_all(image)
    mesh.simplify_by_pairs_remained(40)
    smth = LaplaceSmoother.create_from_mesh(mesh, interpolation="log")
    smth.smooth(converge="steps", steps=1000)
    smth.draw()


class LaplaceSmoother:
    """
    Сглаживаем методом конечных разностей картинку с заданной маской.
    Значения маски фиксированы. Остальные на каждом шаге определяются как среднее между значениями в окрестности.
    """

    EPS = 0.1

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

    def smooth(self, converge='steps', steps=1000, eps=100.0, max_converge_steps=5000):
        """
        Сгладить изображение по комплексу Морса-Смейла.
        Сглаживание производится применением фильтра Лапласа на клетках комплекса Морса.
        В качестве критерия остановки используется либо количество шагов (converge='steps'),
        либо сходимость по сумме значений ('eps'): когда сумма изменяется после шага менее, чем на eps,
        сглаживание завершается.
        :type eps: float
        :type steps: int
        :type max_converge_steps: int
        :param converge:
            Критерий остановки сглаживания.
            'steps' - фиксировать количество шагов;
            'eps' - сходимость по сумме значений.
        :param steps:
            При converge='steps' устанавливает количество шагов сглаживания.
        :param eps:
            При converge='eps' устанавливает критерий сходимости сглаживания.
        :param max_converge_steps:
            Максимальное количество шагов, при сглаживании по методу сходимости по сумме.

        :return:
        """
        converge_allowed_values = ['steps', 'eps']
        if converge not in converge_allowed_values:
            raise AssertionError('Wrong converge value. Allowed values are {0}'.format(converge_allowed_values))
        if converge == 'steps' and steps < 1:
            raise AssertionError('Variable steps takes only positive values')
        if converge == 'eps' and eps <= 0:
            raise AssertionError('Variable eps takes only positive values')

        if converge == 'steps':
            for i in range(steps):
                self.next()
        elif converge == 'eps':
            prev_sum = 0
            curr_sum = np.sum(self.field)  # Просто инициализация нулём
            i = 0
            for i in range(max_converge_steps):
                self.next()
                prev_sum = curr_sum
                curr_sum = np.sum(self.field)
                if np.abs(curr_sum - prev_sum) < eps:
                    break
            if i == max_converge_steps - 1:
                warnings.warn('Maximum converge steps exceeded!')
            else:
                print('Converged in {0} steps'.format(i + 1))

    def next(self):
        """
        Проводим шаг сглаживания.
        """
        # Складываем поле с самим собой, сдвинутым на 1 во всех направлениях.
        # Делим на 4
        # Получаем то же самое, что свёртка с фильтром Лапласа для 4-соседской структуры на торе.
        # Восстанавливаем значения маскированных ячеек.

        new_field = np.zeros(self.field.shape)
        new_field[:self.lx - 1, :self.ly] += self.field[1:self.lx, :self.ly]
        new_field[self.lx - 1, :self.ly] += self.field[0, :self.ly]

        new_field[:self.lx, :self.ly - 1] += self.field[:self.lx, 1:self.ly]
        new_field[:self.lx, self.ly - 1] += self.field[:self.lx, 0]

        new_field[1:self.lx, :self.ly] += self.field[:self.lx - 1, :self.ly]
        new_field[0, :self.ly] += self.field[self.lx - 1, :self.ly]

        new_field[:self.lx, 1:self.ly] += self.field[:self.lx, :self.ly - 1]
        new_field[:self.lx, 0] += self.field[:self.lx, self.ly - 1]

        new_field *= 0.25

        # Значение фиксированных клеток не изменяется
        self.field = np.where(self.mask, self.field, new_field)

    def draw(self, plot_3d=False, antialiased=True):
        """
        Plot smoothed image.
        :param antialiased: Anti-alias.
        :param plot_3d: Plot smoothed image as 3D-surface.
        :return:
        """
        if plot_3d:
            fig = plt.figure()
            x, y = np.meshgrid(range(self.ly), range(self.lx))
            ax = fig.gca(projection='3d')
            ax.plot_surface(x, y, self.field, cmap=cm.gray, linewidth=0, antialiased=antialiased)
            ax.view_init(azim=-30, elev=15)
        else:
            plt.figure()
            cur_plot = plt.imshow(self.field, cmap='gray', origin='lower')
            plt.colorbar(cur_plot)
        plt.show()

    @staticmethod
    def create_from_mesh(mesh, interpolation='linear'):
        """
        Build smoother from Morse-Smale complex.
        1. Put values in critical points
        2. Interpolate values on arcs, fix it.
        3. By finite differences, compute field.
        :param interpolation:
            Interpolation method to set values on arcs.
            'linear', 'log'
        :type mesh: TorusMesh
        """
        lx, ly = mesh.sizeX, mesh.sizeY
        arc_list = mesh.list_arcs()
        arc_list.sort(key=len)
        arc_values_list = []

        # Интерполированием устанавливаем значения на дугах
        for arc in arc_list:
            start_val = mesh._extvalue(arc[0])[0]
            end_val = mesh._extvalue(arc[-1])[0]
            values = np.array([])
            if interpolation == 'linear':
                values = np.linspace(start_val, end_val, len(arc))
            elif interpolation == 'log':
                if np.sign(start_val) == np.sign(end_val):
                    # Если концы дуги одного знака, то интерполируем в лог-масштабе.
                    values = np.geomspace(start_val, end_val, num=len(arc))
                else:
                    # Если в разном, то середину дуги считаем за 0 (с добавкой)
                    # и интерполируем две части дуги.
                    values = np.geomspace(start_val, LaplaceSmoother.EPS * np.sign(start_val), num=len(arc) // 2)
                    if len(arc) % 2 == 1:
                        values = np.append(values, [0.0])
                    values = np.append(values,
                                       np.geomspace(LaplaceSmoother.EPS * np.sign(end_val), end_val, num=len(arc) // 2))

            arc_values_list.append(values)

        # Создаём экземпляр сглаживателя
        a = LaplaceSmoother(lx, ly)
        for arc, arc_val in zip(arc_list, arc_values_list):
            for idx in range(len(arc)):
                x, y = int(mesh.coords(arc[idx])[1]), int(mesh.coords(arc[idx])[0])
                a.set_val(x, y, arc_val[idx])
                a.set_mask(x, y)
        return a
