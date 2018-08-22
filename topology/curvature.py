import cv2
from copy import copy
import numpy as np
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt


def del_crosses(field):
    """
    Удалим пикселы с более чем 3 соседями из множеств Якоби.
    Метод долгий, нужно запускать его отдельно.
    Без его применения идентификация контуров работает некорректно.
    :param field:
    :return:
    """
    print('.', end='')
    img = copy(field)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            cross_sum = 0
            cross_sum += img[i-1, j-1]
            cross_sum += img[i-1, j]
            cross_sum += img[i-1, j+1]
            cross_sum += img[i, j-1]
            cross_sum += img[i, j+1]
            cross_sum += img[i+1, j-1]
            cross_sum += img[i+1, j]
            cross_sum += img[i+1, j+1]
            if cross_sum >= 3:
                img[i, j] = False
    return img


def cmp_contours_binary_image(img):
    """
    Выделяем контуры из бинарного изображения.
    """
    # Выделяем связные компоненты. Их должно быть столько же, сколько и контуров.
    conn_num, conn_image = cv2.connectedComponents(img.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)

    # Теперь работаем с каждой связной компонентой в отдельности.
    conn_images = [(conn_image == i).astype(np.uint8) for i in range(1, conn_num)]
    
    # Выделяем контуры из каждой компоненты
    contours = []
    for im in conn_images:
        im_contours = cv2.findContours(im, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]
        im_contours = [c.reshape(c.shape[0], 2) for c in im_contours]
        
        # Удаляем дубликаты в контуре
        new_contours = []
        for i in range(len(im_contours)):
            c = im_contours[i]
            idxs = (np.argwhere(np.sum((c == np.roll(c, -2, axis=0)).astype(int), axis=1) == 2) + 1).flatten()
            if len(idxs) < 2:
                new_contours.append(c)
            elif len(idxs) % 2 == 0:
                for j in range(len(idxs) // 2):
                    new_contours.append(c[idxs[j]: idxs[j + 1] + 1])
            else:
                new_contours.append(c[:idxs[0] + 1])
                for j in range(1, len(idxs) // 2 + 1):
                    new_contours.append(c[idxs[j]: idxs[j + 1] + 1])

        # Удаляем дублирующиеся контуры
        main_contour = new_contours[0]
        contours.append(main_contour)
        for another_contour in new_contours[1:]:
            intersection = set(map(tuple, main_contour)) & set(map(tuple, another_contour))
            if len(intersection) <= 2:
                contours.append(another_contour)
    
    return [c for c in contours if len(c) >= 3]

    
def cmp_binary_image_curvature(img, plot=False):
    """
    Вычисление средней кривизны изображения.
    Предварительно нужно обработать изображение методом del_crosses,
    иначе идентификация контуров работает некорректно.
    :param img:
    :param plot:
    :return:
    """

    contours = cmp_contours_binary_image(img)
    
    # Костыль: считаем, что контур замкнут, если от начала до конца не более 15 пикселей.
    for i in range(len(contours)):
        c = contours[i]
        if Point(c[-1]).distance(Point(c[0])) <= 15:
            contours[i] = np.vstack([c, [c[0]]])
    
    contours = [np.array(list(LineString(c).simplify(tolerance=np.sqrt(2)).coords)) for c in contours]  # Упрощаем контуры Дугласом-Пекером
    
    if plot:
        plt.figure()
        for c in contours:
            plt.plot(np.transpose(c)[0], np.transpose(c)[1], '-r', lw=1)

    total_curv = 0
    total_length = 0
    for c in contours:
        total_length += LineString(c).length  # Добавляем длину контура к общей длине
        for i in range(1, len(c) - 1):
            p1, p2, p3 = c[i-1], c[i], c[i+1]  # Точки кривой
            v1, v2 = (p2[0] - p1[0], p2[1] - p1[1]), (p3[0] - p2[0], p3[1] - p2[1])  # Векторы
            angle = np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))
            total_curv += angle  # Добавляем к общей кривизне    
        # Проверяем контур на замкнутость
        if (c[0][0] == c[-1][0]) and (c[0][1] == c[-1][1]):
            # Контур замкнут            
            p1, p2, p3 = c[-2], c[0], c[1]  # Точки кривой
            v1, v2 = (p2[0] - p1[0], p2[1] - p1[1]), (p3[0] - p2[0], p3[1] - p2[1])  # Векторы
            angle = np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))
            total_curv += angle  # Добавляем к общей кривизне
        
    print('Number of contours: {contours}\nTotal curvature: {curv}\nTotal length: {length}\nMean curvature: {mean_curv}\n'.
          format(contours=len(contours), curv=total_curv, length=total_length, mean_curv=total_curv / total_length))
    return total_curv / total_length


def cmp_binary_image_curvature_sphere(sg, img, plot=False):
    """
    Вычисление средней кривизны изображения на сфере.
    Предварительно нужно обработать изображение методом del_crosses,
    иначе идентификация контуров работает некорректно.
    :param img:
    :param plot:
    :return:
    """
    contours = cmp_contours_binary_image(img)

    # Костыль: считаем, что контур замкнут, если от начала до конца не более 15 пикселей.
    for i in range(len(contours)):
        c = contours[i]
        if Point(c[-1]).distance(Point(c[0])) <= 15:
            contours[i] = np.vstack([c, [c[0]]])

    # Упрощаем контуры Дугласом-Пекером
    contours = [np.array(list(LineString(c).simplify(tolerance=np.sqrt(2)).coords)) for c in contours]

    if plot:
        plt.figure()
        for c in contours:
            plt.plot(np.transpose(c)[0], np.transpose(c)[1], '-r', lw=1)

    total_curv = 0
    total_length = 0
    for c in contours:
        c = sg._convert_to_spherical(c)  # Переводим в сферические
        total_length += sg.sphere_contour_distance(c)  # Добавляем длину контура к общей длине
        for i in range(1, len(c) - 1):
            p1, p2, p3 = c[i-1], c[i], c[i+1]  # Точки кривой
            angle = sg.sphere_angle(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
            total_curv += angle  # Добавляем к общей кривизне
        # Проверяем контур на замкнутость
        if (c[0][0] == c[-1][0]) and (c[0][1] == c[-1][1]):
            # Контур замкнут
            p1, p2, p3 = c[-2], c[0], c[1]  # Точки кривой
            angle = sg.sphere_angle(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
            total_curv += angle  # Добавляем к общей кривизне

    print('Number of contours: {contours}\nTotal curvature: {curv}\nTotal length: {length}\nMean curvature: {mean_curv}\n'.
          format(contours=len(contours), curv=total_curv, length=total_length, mean_curv=total_curv / total_length))
    return total_curv / total_length


def cmp_binary_image_curvature_sphere_quad(sg, img, lat0, lat1, lon0, lon1, plot=False):
    """
    Вычисление средней кривизны изображения на заданном квадранте сферы.
    Предварительно нужно обработать изображение методом del_crosses,
    иначе идентификация контуров работает некорректно.
    :param img:
    :param plot:
    :return:
    """
    idx_lat0, idx_lat1, idx_lon0, idx_lon1 = sg.convert_index(lat0, lat1, lon0, lon1)
    img_copy = copy(img)

    # Стираем всё за границами окна
    img_copy[:, idx_lon0] = False
    img_copy[:, idx_lon1:] = False
    img_copy[:idx_lat0, :] = False
    img_copy[idx_lat1:, :] = False

    if plot:
        plt.figure()
        plt.imshow(img_copy)
        plt.plot([idx_lon0, idx_lon1, idx_lon1, idx_lon0], [idx_lat0, idx_lat0, idx_lat1, idx_lat1])
        plt.show()

    return cmp_binary_image_curvature_sphere(sg, img_copy, plot)
