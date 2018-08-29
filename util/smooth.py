import numpy as np
import functools
from copy import copy
import scipy.stats

longitude_large = np.linspace(-180, 180, 1441, endpoint=True)
latitude_large = np.linspace(-90, 90, 721, endpoint=True)

longitude_small = np.linspace(-180, 180, 361, endpoint=True)
latitude_small = np.linspace(-90, 90, 181, endpoint=True)

longitude_large_2 = np.linspace(-179.875, 179.875, 1440, endpoint=True)
latitude_large_2 = np.linspace(-89.875, 89.875, 720, endpoint=True)

kernel_size_x = np.tile(np.abs(np.sin(latitude_large_2 * np.pi / 180)) + 0.5, (1440, 1)).T
int_kernel_size_x = (15 * kernel_size_x).astype(np.int) + np.ones(kernel_size_x.shape, dtype=np.int)
int_kernel_size_y = np.empty(kernel_size_x.shape, dtype=np.int)
int_kernel_size_y.fill(np.min(int_kernel_size_x))


@functools.lru_cache(maxsize=100)
def _get_gauss_kernel(rx, ry):
    """
    Дискретное гауссовское ядро заданного радиуса
    """
    if (rx == 0) or (ry == 0):
        return np.ones((1, 1))

    x, y = np.meshgrid(np.arange(-rx, rx + 1, 1),
                       np.arange(-ry, ry + 1, 1))
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x;
    pos[:, :, 1] = y
    rv = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[rx * rx, 0], [0, ry * ry]])  # !TODO: Посчитать коварианс
    result = rv.pdf(pos)
    return result / np.sum(result)  # Нормируем


def _variable_blur(data, kernel_size_x, kernel_size_y):
    """ Blur with a variable window size
        Width and height of kernel are different.
        (on sphere, height of kernel is constant, but width is different)
    Parameters:
      - data: 2D ndarray of floats or integers
      - kernel_size_x: 2D ndarray of integers, same shape as data (x-radius of kernel)
      - kernel_size_y: 2D ndarray of integers, same shape as data (y-radius of kernel)
    Returns:
      2D ndarray
    """
    size_i, size_j = data.shape
    max_kernel_size = np.max([np.max(kernel_size_x), np.max(kernel_size_y)])
    # Добавляем нули к данным по краям
    ext_data = np.zeros((size_i + 2 * max_kernel_size,
                         size_j + 2 * max_kernel_size))
    ext_data[max_kernel_size: -max_kernel_size, max_kernel_size: -max_kernel_size] = data[:]

    # Расширяем для корректного сглаживания (продолжением за склейку)
    # Склеиваем по долготе, продлеваем по широте
    ext_data[:max_kernel_size, max_kernel_size: -max_kernel_size] = data[0, :]
    ext_data[-max_kernel_size:, max_kernel_size: -max_kernel_size] = data[-1, :]
    ext_data[:, :max_kernel_size] = ext_data[:, -2 * max_kernel_size:-max_kernel_size]
    ext_data[:, -max_kernel_size:] = ext_data[:, max_kernel_size:2 * max_kernel_size]

    data_blurred = np.zeros(data.shape)
    for i in range(size_i):
        print('.'.format(i, size_i), end='')
        for j in range(size_j):
            sigma_x = kernel_size_x[i, j]
            sigma_y = kernel_size_y[i, j]

            # Сворачиваем с Гауссом
            res = np.dot((ext_data[max_kernel_size + i - sigma_x: max_kernel_size + i + sigma_x + 1,
                          max_kernel_size + j - sigma_y: max_kernel_size + j + sigma_y + 1]).flatten(),
                         (_get_gauss_kernel(sigma_x, sigma_y)).flatten())
            data_blurred[i, j] = res
    print('Finished')
    return data_blurred


def _last_full_index(data):
    return np.max(np.where(np.sum(np.isnan(data), axis=1) == 0))


def _extend_nans(data):
    # Выбираем последнюю целую широту, заполняем дальнейшие строчки так же.
    result = copy(data)

    # Смотрим последнюю строку, в которой есть хотя бы половина значений.
    last_half_index = np.max(np.where(np.sum(np.isnan(data), axis=1) <= data.shape[1] / 2))

    result[:last_half_index, :] = sew(result[:last_half_index, :])

    # Далее, просто доопределяем как значения в последней строке.
    for i in range(last_half_index, data.shape[0]):
        result[i, :] = result[last_half_index - 1, :]
    return result, last_half_index


def smooth_data(data, cut=False):
    """
    Сглаживаем данные на сфере.
    Параметры сглаживания заданы в начале скрипта.
    :param cut: обрезать дополнительные данные.
    :param data:
    :return:
    """
    # Закрываем Nan's
    result, last_half_nan = _extend_nans(data)
    print('Last half value: {0}'.format(last_half_nan))

    result = _variable_blur(result, kernel_size_x=int_kernel_size_x, kernel_size_y=int_kernel_size_y)

    # Отрезаем обратно
    if cut:
        result[last_half_nan + 1:, :] = np.nan
    return result


def _auto_step(source):
    print('Step: {0} nan values'.format(np.sum(np.isnan(source))))
    result = copy(source)
    nan_x, nan_y = np.where(np.isnan(source))
    for i in range(len(nan_x)):
        # Топология цилиндра, через короткую сторону не склеиваем
        if nan_x[i] == 0:
            result[0, nan_y[i]] = np.nanmean([source[0, nan_y[i] - 1],
                                              source[1, nan_y[i]],
                                              source[0, nan_y[i] + 1]])
        elif nan_x[i] == source.shape[0] - 1:
            result[nan_x[i], nan_y[i]] = np.nanmean([source[nan_x[i] - 1, nan_y[i]],
                                                     source[nan_x[i], nan_y[i] - 1],
                                                     source[nan_x[i], (nan_y[i] + 1) % source.shape[1]]])
        else:
            result[nan_x[i], nan_y[i]] = np.nanmean([source[nan_x[i] - 1, nan_y[i]],
                                                     source[(nan_x[i] + 1) % source.shape[0], nan_y[i]],
                                                     source[nan_x[i], nan_y[i] - 1],
                                                     source[nan_x[i], (nan_y[i] + 1) % source.shape[1]]])
    return result


def sew(source):
    """
    Зашиваем данные клеточным автоматом.
    :param source: Исходное изображение.
    :return:
    """
    result = copy(source)
    while np.sum(np.isnan(result)) > 0:
        result = _auto_step(result)
    return result


def test():
    import matplotlib.pyplot as plt
    from importers import ozone_importer
    oz_importer = ozone_importer.OzoneImporter()
    ozone_maps = oz_importer.get_fields(start_date='2016/01/21', finish_date='2016/02/03', large=True)
    # test_map = np.nanmean(ozone_maps[:3], axis=0)
    test_map = ozone_maps[0]
    plt.matshow(test_map, origin='lower', cmap='jet')
    plt.show()
    result = smooth_data(test_map)
    plt.matshow(result, origin='lower', cmap='jet')
    plt.show()

