import numpy as np
import topology.curvature


class SphericalGeometry:
    def __init__(self, radius, lons, lats):
        """
        Создать класс для вычисления функций геометрии на сфере.
        Индексация широт -90..90
        Индексация долгот -180..180
        :param radius: радиус сферы
        :param lons: количество разбиений по долготам
        :param lats: количество разбиений по широтам
        :return:
        """
        self.lons = lons
        self.lats = lats
        self.r = radius

        np.linspace(-89.875, 89.875, 720, endpoint=True)

        # Градусная сетка на сфере lats x lots
        xx, yy = np.meshgrid([2 * np.pi / lons] * lons,
                             (np.linspace(-90, 90, lats + 1, endpoint=True) + 180/(lats * 2))[:-1] * np.pi / 180)
        dxx = np.zeros(xx.shape)
        dyy = np.zeros(yy.shape)
        dxx.fill(2 * np.pi / lons)
        dyy.fill(np.pi / lats)

        # Считаем распределение по площадив зависимостиот координат квадранта на сфере
        self.area_grid = self.unit_area_eps(xx, yy, dxx, dyy)

    def sphere_unit_area(self, phi_1, phi_2, th_1, th_2):
        """
        Площадь участка сферы, ограниченного двумя меридианами и двумя параллелями.
        :param r: Радиус сферы
        :param phi_1: Первый азимутальный угол
        :param phi_2: Второй азимутальный угол
        :param th_1: Первый зенитный угол
        :param th_2: Второй зенитный угол
        :return:
        """
        return self.r**2 * (phi_2 - phi_1) * (np.sin(th_2) - np.sin(th_1))

    def unit_area_eps(self, phi, th, dphi, dth):
        """
        Площадь участка сферы радиуса r, ограниченного меридианами phi+-dphi
        и параллелями th+-dth.
        :param r: Радиус сферы
        :param phi: Центр участка по азимуту
        :param th: Центр участка по зениту
        :param dphi: Радиус окрестности по азимуту
        :param dth: Радиус окрестности по зениту
        :return:
        """
        return self.sphere_unit_area(phi - dphi / 2, phi + dphi / 2, th - dth/2, th + dth/2)

    def curve_perimeter(self, binary_image):
        """
        Периметр кривых на бинарном изображении, спроецированном на сферу.
        :param binary_image: Бинарное изображение
        :return:
        """
        return np.linalg.multi_dot([binary_image.flatten(), self.area_grid.flatten()])

    def curve_perimeter_quad(self, binary_image, lat0, lat1, lon0, lon1):
        """
        Периметр кривых на бинарном изображении, спроецированном на сферу
        в заданном квадранте сферы
        :param binary_image: Бинарное изображение
        :param lat0:
        :param lat1:
        :param lon0:
        :param lon1:
        :return:
        """
        idx_lat0 = int((1 + lat0 / 90) * 0.5 * self.lats)
        idx_lat1 = int((1 + lat1 / 90) * 0.5 * self.lats)
        idx_lon0 = int((1 + lon0 / 180) * 0.5 * self.lons)
        idx_lon1 = int((1 + lon1 / 180) * 0.5 * self.lons)
        return np.linalg.multi_dot([binary_image[idx_lat0:idx_lat1, idx_lon0:idx_lon1].flatten(),
                                    self.area_grid[idx_lat0:idx_lat1, idx_lon0:idx_lon1].flatten()])

    def mean_curvature(self, binary_image):
        """
        Средняя кривизна линий на сфере.
        :param binary_image: Бинарное изображение
        :return:
        """
        curvature_map = topology.curvature.get_curvature_map(binary_image)
        return np.linalg.multi_dot([curvature_map.flatten(), self.area_grid.flatten()]) /\
               self.curve_perimeter(binary_image)

    def mean_curvature_quad(self, binary_image, lat0, lat1, lon0, lon1):
        """
        Средняя кривизна линий на бинарном изображении, спроецированном на сферу
        в заданном квадранте сферы
        :param binary_image: Бинарное изображение
        :param lat0:
        :param lat1:
        :param lon0:
        :param lon1:
        :return:
        """
        idx_lat0 = int((1 + lat0 / 90) * 0.5 * self.lats)
        idx_lat1 = int((1 + lat1 / 90) * 0.5 * self.lats)
        idx_lon0 = int((1 + lon0 / 180) * 0.5 * self.lons)
        idx_lon1 = int((1 + lon1 / 180) * 0.5 * self.lons)
        curvature_map = topology.curvature.get_curvature_map(binary_image[idx_lat0:idx_lat1, idx_lon0:idx_lon1])
        return np.linalg.multi_dot([curvature_map.flatten(), self.area_grid[idx_lat0:idx_lat1, idx_lon0:idx_lon1].flatten()]) /\
               self.curve_perimeter_quad(binary_image, lat0, lat1, lon0, lon1)


def _test():
    import matplotlib.pyplot as plt

    s = SphericalGeometry(1, 1440, 720)
    print(s.area_grid.sum())
    plt.matshow(s.area_grid)
    plt.colorbar()
    plt.show()


def _test2():
    s = SphericalGeometry(1, 1440, 720)
    s.curve_perimeter_quad(None, 30, 90, -180, 180)