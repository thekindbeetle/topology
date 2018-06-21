import numpy as np
import skimage.morphology
import skimage.filters
import skimage.measure
import morse.triangmesh
from copy import copy


def binary_jacobi_set(field1, field2, threshold=0.7):
    """
    Приближённое вычисление множества Якоби
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


def precise_binary_jacobi_set(field1, field2, conditions='plain'):
    """
    Точное вычисление множества Якоби по дискретной модели.
    :param conditions: Граничные условия.
    :param field1: Первое поле.
    :param field2: Второе поле.
    :return: Множество Якоби в виде бинарного изображения.
    """
    tr_mesh = morse.triangmesh.TriangMesh.build_all(field1, field2, conditions=conditions)

    # Переводим геометрическую модель в изображение.
    jacobi_mask = np.zeros((tr_mesh.sizeX, tr_mesh.sizeY))
    for e in tr_mesh.jacobi_set:
        jacobi_mask[tr_mesh.coordy(e[0])][tr_mesh.coordx(e[0])] = 1
        jacobi_mask[tr_mesh.coordy(e[1])][tr_mesh.coordx(e[1])] = 1
    jacobi_mask[0] = 0
    jacobi_mask[-1] = 0
    jacobi_mask[:, 0] = 0
    jacobi_mask[:, -1] = 0
    jacobi_mask = skimage.morphology.skeletonize(jacobi_mask)
    return jacobi_mask


def _compute_curve_persistence(source_field, components_map, component_num):
    """
    Персистентность кривой бинарного множества Якоби.
    :param source_field: Исходное поле.
    :param components_map: Карта компонент связности.
    :param component_num: Номер компоненты связности.
    :return: персистентность компоненты связности относительно функции source_field.
    """
    # Вычисляем множество индексов изображения кривой
    indexes = np.argwhere(components_map == component_num)
    # Ищем значения поля на кривой Якоби.
    values = dict([(tuple(idx), source_field[idx[0], idx[1]]) for idx in indexes])
    # Разность макс. и мин. значений первого поля на кривой Якоби - персистентность кривой.
    pers = max(values.values()) - min(values.values())
    return pers


def simplify_jacobi_set(jacobi_field, src_field, persistence_level):
    """
    Персистентное упрощение множества Якоби.
    :param jacobi_field: бинарное изображение множества Якоби.
    :param src_field: исходное поле (одно из исходных полей, по нему вычисляется персистентность контуров).
    :param persistence_level: уровень, по которому производится упрощение.
    :return: упрощенное поле Якоби в виде бинарного изображения.
    """
    components = skimage.measure.label(jacobi_field, neighbors=8)  # Выделяем связные компоненты.
    comp_nums = range(1, components.max())  # Количество связных компонент.
    persistence_comp = dict([(component_num, _compute_curve_persistence(src_field, components, component_num))
                             for component_num in comp_nums]) # Вычисляем персистентность контуров.
    # Фильтруем компоненты с маленькой персистентностью.
    low_persistence_components = [component_num for component_num in comp_nums
                                  if persistence_comp[component_num] < persistence_level]
    result = copy(components)
    for comp_num in low_persistence_components:
        result[components==comp_num] = 0

    return result

