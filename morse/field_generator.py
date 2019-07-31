import copy
import numpy as np
import numpy.random
import scipy.stats
import astropy.io.fits as fits
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

ALLOWABLE_CONDITIONS = ('torus', 'plain')


def _check_conditions(text):
    if text not in ALLOWABLE_CONDITIONS:
        raise AssertionError('Неверно заданы граничные условия.\n'
                             'Допустимы следующие варианты: {0}'.format(ALLOWABLE_CONDITIONS))


def gen_gaussian_sum_rectangle(rows_num, cols_num, centers, sigma):
    """
    Матрица, являющаяся суммой гауссиан (на прямоугольнике)
    :param rows_num: количество строк матрицы
    :param cols_num: количество столбцов матрицы
    :param centers: список центров (не обязательно целочисленных, могут даже лежать за пределами матрицы!)
    :param sigma: ширина гауссиан
    """
    field = np.zeros((rows_num, cols_num))
    for center in centers:
        for i in range(rows_num):
            for j in range(cols_num):
                field[i, j] += scipy.stats.multivariate_normal.pdf((i, j), mean=center, cov=((sigma, 0), (0, sigma)))
    print("Field {0}x{1} generated as sum of {2} gaussians".format(rows_num, cols_num, len(centers)))
    return field


def gen_gaussian_sum_torus(rows_num, cols_num, centers, sigma, log=True):
    """
    Матрица, являющаяся суммой гауссиан на торе.
    Считаем значения гауссиан за пределами 3-сигма нулевыми.
    :param rows_num: количество строк матрицы
    :param cols_num: количество столбцов матрицы
    :param centers: список центров (не обязательно целочисленных, могут даже лежать за пределами матрицы!)
    :param sigma: ширина гауссиан
    :param log: Включить текстовый вывод
    :return:
    """
    if log:
        print('Generation started...')
    field = np.zeros((rows_num, cols_num))
    sigma3 = int(sigma * 3)
    gaussian = np.zeros((sigma3 * 2 + 1, sigma3 * 2 + 1))
    for i in range(-sigma3, sigma3 + 1):
        for j in range(-sigma3, sigma3 + 1):
            if i ** 2 + j ** 2 <= sigma3:
                gaussian[i, j] += scipy.stats.multivariate_normal.pdf((i, j), mean=(0, 0), cov=((sigma, 0), (0, sigma)))
    if log:
        print('Gaussian values computed, summing...', end='')

    checkpoints_num = 20
    current_checkpoint = 0

    for i in range(len(centers)):
        center = centers[i]

        if log and i > len(centers) * current_checkpoint / checkpoints_num:
            current_checkpoint += 1
            print('.', end='')

        for i in range(-sigma3, sigma3 + 1):
            for j in range(-sigma3, sigma3 + 1):
                field[(int(center[0]) + i) % rows_num, (int(center[1]) + j) % cols_num] += gaussian[i, j]

    print('\nGeneration completed.')
    return field


def gen_sincos_field(rows_num, cols_num, kx, ky):
    """
    Матрица функции sin (x * kX) + cos (y * kY)
    :param rows_num: количество строк матрицы
    :param cols_num: количество столбцов матрицы
    :param kx: коэффициент по X
    :param ky: коэффициент по Y
    :return:
    """
    func = lambda x, y: np.sin(x * kx) + np.cos(y * ky)
    field = np.zeros((rows_num, cols_num))
    for i in range(rows_num):
        for j in range(cols_num):
            field[i, j] = func(i, j)
    return field


def gen_field_from_file(filename, filetype='bmp', conditions='torus', compression=10, perturb_data=True):
    """
    Построить сетку по изображению.
    В узлах значения яркости в пикселе.
    При склейке в тор изображение отражается вправо и вверх с заданным коэффициентом сжатия.
    :param filetype: Тип файла (24bit-BMP, FITS или ASC).
    :param conditions: Граничные условия. 'torus' для склейки в тор, 'plain' — без склейки.
    :param filename: Файл с изображением.
    :param compression: Коэффициент сжатия картинки при склейке в тор. По умолчанию 10 (сжатие в десять раз).
    :param perturb_data: Возмущение изображения (по умолчанию — True).
    :return:
    """
    allowed_file_types = ('bmp', 'fits', 'asc')

    if filetype not in allowed_file_types:
        raise AssertionError('Данный формат файла не поддерживается\n'
                             'Допустимые форматы: {0}'.format(allowed_file_types))
    _check_conditions(conditions)
    is_torus_conditions = (conditions == 'torus')
    if is_torus_conditions and compression <= 0:
        raise AssertionError('Коэффициент сжатия должен быть положительным!')

    if filetype == 'bmp':
        with Image.open(filename) as bmp_image:
            image = bmp_image.convert('L')
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.transpose(Image.ROTATE_180)

            if is_torus_conditions:
                extension_x = int(image.size[0] / compression)
                extension_y = int(image.size[1] / compression)
                # Расширяем изображение
                new_image = Image.new('L', (image.size[0] + extension_x, image.size[1] + extension_y))
                new_image.paste(image, (0, 0))
                horizontal_image = image.resize((extension_x, image.size[1])).transpose(Image.FLIP_LEFT_RIGHT)
                new_image.paste(horizontal_image, (image.size[0], 0))
                vertical_image = image.resize((image.size[0], extension_y)).transpose(Image.FLIP_TOP_BOTTOM)
                new_image.paste(vertical_image, (0, image.size[1]))
                rotated_image = image.resize((extension_x, extension_y)).transpose(Image.ROTATE_180)
                new_image.paste(rotated_image, image.size)
            else:
                new_image = image
            field = np.array(new_image, dtype=float)

            if perturb_data:
                perturb(field)

            return field
    elif filetype == 'fits':
        import sunpy.map
        # I cannot read compressed image usual way.
        mp = sunpy.map.Map(filename)
        image = mp.data
        if is_torus_conditions:
            rotated_image = np.rot90(image, k=2)
            rotated_image = rotated_image[::compression, ::compression]
            horizontal_image = image[::-1, ::compression]
            vertical_image = image[::compression, ::-1]
            image = np.hstack((np.vstack((image, vertical_image)), np.vstack((horizontal_image, rotated_image))))
        if perturb_data:
            perturb(image)
        return image
    elif filetype == 'asc':
        grid = np.loadtxt(filename, skiprows=6)
        if is_torus_conditions:
            grid = np.hstack((np.vstack((grid[:, :], grid[::-compression, :])),
                              np.vstack((grid[:, ::-compression], grid[::-compression, ::-compression]))))
        if perturb_data:
            perturb(grid)
        return grid


def apply_conditions_to_field(field, conditions='torus', compression=10):
    _check_conditions(conditions)
    if conditions == 'torus':
        print(field.shape)
        bottom_image = field[:, ::-compression]
        right_image = field[::-compression, :]
        rotated_image = np.rot90(field, k=2)
        rotated_image = rotated_image[::compression, ::compression]
        newfield = np.hstack((np.vstack((field, right_image)), np.vstack((bottom_image, rotated_image))))
        return newfield
    else:
        return copy.deepcopy(field)


def resize(field, multipliers=(10, 10), sigma=(25, 25), gauss=True, conditions='plain'):
    """
    Расширение массива с последующим сглаживанием Гауссовским фильтром.
    :param gauss: сглаживание Гауссом.
    :param conditions: граничные условия, накладываемые при применении фильтра.
    :param field: Входные данные.
    :param multipliers: Коэффициенты расширения изображения по осям.
    :param sigma: Ширина Гауссовского фильтра по осям.
    :return:
    """
    _check_conditions(conditions)
    new_field = np.concatenate(
        [np.concatenate([np.full(multipliers, field[i, j], dtype=field.dtype)
                         for i in range(field.shape[0])], axis=0)
         for j in range(field.shape[1])], axis=1)
    if gauss:
        if conditions == 'plain':
          new_field = gaussian_filter(new_field, sigma)
        elif conditions == 'torus':
          new_field = gaussian_filter(new_field, sigma, mode='wrap')
    return new_field


def perturb(field, eps=0.000001):
    """
    Возмущение матрицы. На выходе - матрица, сохраняющая все топологические особенности, все значения которой различны.
    :param field: Матрица (numpy.ndarray)
    :param eps: Значения, отличающиеся меньше чем на eps, считаем равными
    :return:
    """
    flat = sorted(field.flatten())
    try:
        dif = min([np.abs(flat[i + 1] - flat[i]) for i in range(len(flat) - 1) if np.abs(flat[i + 1] - flat[i]) > eps])
        print("Minimal difference is {0}".format(dif))
        lx = field.shape[0]
        ly = field.shape[1]
        for i in range(lx):
            for j in range(ly):
                field[i, j] += dif * (i + lx * j) / (2 * lx * ly)
        print("Field perturbed.")
    except ValueError:
        print("Error: field values differ less than epsilon value = {0}. Field was not perturbed.".format(eps))
    return field


def perturb2(field):
    lx = field.shape[0]
    ly = field.shape[1]
    field += numpy.random.rand(lx, ly)