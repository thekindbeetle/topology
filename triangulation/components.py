import numpy as np

from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from generators import poisson
from triangulation.persistence.persistence import Persistence
from triangulation.persistence.filtration import Filtration


def phi_empirical(x, y, plot=True, log=False, label="Empirical data"):
    """
    График зависимости количества компонент связности от времени (нормированный)
    для набора точек на квадрате [0,1]x[0,1].
    :param x: Список координат x точек
    :param y: Список координат y точек
    :param plot: Отображение графика
    :param log: Вывод сообщений
    :param label: Метка на графике
    :return: lambda-функция.
    """
    if len(x) != len(y):
        raise RuntimeError('Количество координат x и y должно быть одинаковым!')

    pnum = len(x)
    filt = Filtration.from_points(x, y)
    pers = Persistence(filt)

    def comp_num(level):
        return len([x for x in pers._compDeathTimes if x >= level])

    def f(level):
        return comp_num(level / np.sqrt(pnum)) / (pnum - 1)

    if plot:
        p = np.linspace(0, np.sqrt(2), 1000)
        plt.plot(p, list(map(f, p)), '-', label='{emp}: {pnum} points'.format(emp=label, pnum=pnum))

    return f


def _phi_precise(pnum, evnum=200, plot=True, loglevel=0):
    """
    График зависимости количества компонент связности от времени (нормированный)
    для пуассоновского процесса с заданной интенсивностью.
    :param pnum: Количество точек на квадрате [0,1]x[0,1] (интенсивность процесса)
    :param evnum: Количество событий
    :param plot: Отображение графика
    :param loglevel: Подробность логирования
    :return: функция.
    """

    # Список функций phi_emp для различных реализаций Пуассоновского процесса.
    phi_emp = []

    for i in range(evnum):
        x, y = poisson.poisson_homogeneous_point_process(pnum, 1, log=(loglevel > 1))
        phi_emp.append(phi_empirical(x, y, plot=False, log=False))

    def f(level):
        return np.mean(list(map(lambda g: g(level), phi_emp)))

    if plot:
        p = np.linspace(0, np.sqrt(2), 1000)
        plt.plot(p, list(map(f, p)), '-', label='{0} points'.format(pnum))

    return f


def _phi_approx(pnum, evnum=200, plot=True, loglevel=0):
    """
    График зависимости количества компонент связности от времени (нормированный)
    для пуассоновского процесса с заданной интенсивностью.
    Считаем приближённо (значения считаются в точках на отрезке [0,sqrt(2)]).
    Интеграл считается по этим же точкам, т. е. приближённо,
    значение немного меньше реального из-за выпуклости функции
    и исключения хвоста.
    :param pnum: Количество точек на квадрате [0,1]x[0,1] (интенсивность процесса)
    :param evnum: Количество событий
    :param plot: Отображение графика
    :param loglevel: Подробность логирования
    :return: функция.
    """
    p = np.linspace(0, np.sqrt(2), 1000)
    values = np.zeros(len(p))

    for i in range(evnum):
        x, y = poisson.poisson_homogeneous_point_process(pnum, 1, log=(loglevel > 1))
        g = phi_empirical(x, y, plot=False, log=False)
        for i in range(len(p)):
            values[i] += g(p[i])

    values /= evnum

    f = interp1d(p, values, fill_value="extrapolate")

    if plot:
        plt.plot(p, values, '-', label='{0} points'.format(pnum))

    return f


def _phi_simple(pnum, evnum=200, plot=True, loglevel=0):
    """
    График зависимости количества компонент связности от времени (нормированный)
    для пуассоновского процесса с заданной интенсивностью.
    Здесь фиксировано количество точек в квадрате, т.е. оно не является случайной величиной, как в остальных методах.
    Подходит для pnum, начиная с 20.
    Метод работает быстрее остальных.
    :param pnum: Количество точек на квадрате [0,1]x[0,1] (интенсивность процесса)
    :param evnum: Количество событий
    :param plot: Отображение графика
    :param loglevel: Подробность логирования
    :return: функция.
    """
    death_comps = [[] for i in range(pnum - 1)]

    for i in range(evnum):
        # Количество точек фиксировано
        x, y = poisson.poisson_homogeneous_point_process(pnum, 1, log=False, fixed_rate=True)
        filt = Filtration.from_points(x, y)
        pers = Persistence(filt)
        pers._compDeathTimes.sort(reverse=True)
        for j in range(pnum - 1):
            death_comps[j].extend([pers._compDeathTimes[j]])

    death_comps_tr = np.transpose(death_comps)

    def b0(level, idx):
        return len([_ for _ in death_comps_tr[idx] if _ > level]) / (pnum - 1)

    def cum_b0(level):
        return sum([b0(level, i) for i in range(evnum)]) / evnum

    def f(level):
        return cum_b0(level / np.sqrt(pnum))

    if plot:
        p = np.linspace(0, np.sqrt(2), 1000)
        plt.plot(p, list(map(f, p)), '-', label='{0} points'.format(pnum))

    return f


def phi(pnum, evnum=200, plot=True, loglevel=0, method='simple'):
    """
    График зависимости количества компонент связности от времени (нормированный)
    для пуассоновского процесса с заданной интенсивностью.
    :param pnum: Количество точек на квадрате [0,1]x[0,1] (интенсивность процесса)
    :param evnum: Количество событий
    :param plot: Отображение графика
    :param loglevel: Подробность логирования
    :param method: Метод вычисления: 'simple', 'approx' или 'precise'.
    :return: lambda-функция.
    """
    if method == 'simple':
        return _phi_simple(pnum, evnum=evnum, plot=plot, loglevel=loglevel)
    elif method == 'approx':
        return _phi_approx(pnum, evnum=evnum, plot=plot, loglevel=loglevel)
    elif method == 'precise':
        return _phi_precise(pnum, evnum=evnum, plot=plot, loglevel=loglevel)
    else:
        raise RuntimeError('Computation method \"{method}\"is not available'.format(method=method))


def I(pnum, evnum=200, plot=False, loglevel=0, method='simple'):
    """
    Интеграл по области определения функции Ф(n).
    """
    f = phi(pnum, evnum=evnum, plot=plot, loglevel=loglevel)

    return integrate.quad(f, 0, np.sqrt(2))


def I_empirical(x, y, plot=False, loglevel=0):
    """
    Интеграл по области определения функции Ф^emp(n).
    """
    f = phi_empirical(x, y, plot=plot, log=(loglevel > 0))

    return integrate.quad(f, 0, np.sqrt(2))


def classify(x, y, evnum=200, plot=True, log=False, method='simple'):
    """
    Провести классификацию точечных данных на основе сравнения Ф-функций.
    :param x: Список координат x точек
    :param y: Список координат y точек
    :param evnum: количество смоделированных Пуассоновских процессов
    :param plot: отображение графика
    :param log: Вывод сообщений
    """
    if len(x) != len(y):
        raise RuntimeError('Количество координат x и y должно быть одинаковым!')
    pnum = len(x)

    if plot:
        plt.figure()
        plt.plot(x, y, 'ok')
        plt.figure()

    # Find I and I^emp values
    I1 = I(pnum, evnum=evnum, loglevel=0, plot=True, method=method)
    I2 = I_empirical(x, y, loglevel=0, plot=True)

    print("Integral for experiment: {0}".format(I1))
    print("Integral for Poisson: {0}".format(I2))


def test():
    import rpy2.robjects as robjects

    robjects.r('library("spatstat")')
    x = robjects.r('lansing$x')
    y = robjects.r('lansing$y')
    classify(x, y, evnum=20, plot=True, log=True, method='approx')
    plt.legend()
    plt.show()


def test2():
    phi(10000, 20, plot=True, method='simple')
    plt.legend()
    plt.show()
