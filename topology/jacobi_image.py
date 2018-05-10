import numpy as np
import skimage.morphology
import skimage.filters


def binary_jacobi_set(field1, field2, threshold=0.7):
    """

    :param field1: Первое поле.
    :param field2: Второе поле.
    :param threshold: Порог бинаризации изображения. Подбирается вручную в зависимости от данных.
    :return:
    """
    field1 = field1 / np.nansum(field1) * field1.size  # Нормируем поля
    field2 = field2 / np.nansum(field2) * field2.size  # Нормируем поля

    # Считаем градиенты в каждой точке.
    grad1 = np.dstack((skimage.filters.sobel_h(field1), skimage.filters.sobel_v(field1)))
    grad2 = np.dstack((skimage.filters.sobel_h(field2), skimage.filters.sobel_v(field2)))

    # Считаем логарифм градиентной меры.
    grad = np.log(np.abs(np.cross(grad1, grad2)))

    # Находим множество Якоби как ноль градиентной меры.
    jac = skimage.filters.sobel(grad)

    # Чистим его при помощи последовательной дилатации и эрозии.
    jac[np.isnan(jac)] = 0
    jac = (jac > threshold)
    jac = skimage.morphology.erosion(skimage.morphology.erosion(skimage.morphology.dilation(jac)))

    # Делаем линии тонкими засчёт скелетонизации.
    jac = skimage.morphology.skeletonize(jac)
    return jac
