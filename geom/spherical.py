import numpy as np
import topology.curvature
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt


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
        self.lonslist = np.linspace(-180, 180, lons + 1, endpoint=True)
        self.latslist = np.linspace(-90, 90, lats + 1, endpoint=True)
        self.r = radius

        # Градусная сетка на сфере lats x lots
        xx, yy = np.meshgrid([2 * np.pi / lons] * lons,
                             (np.linspace(-90, 90, lats + 1, endpoint=True) + 180/(lats * 2))[:-1] * np.pi / 180)
        dxx = np.zeros(xx.shape)
        dyy = np.zeros(yy.shape)
        dxx.fill(2 * np.pi / lons)
        dyy.fill(np.pi / lats)

        # Считаем распределение по площади в зависимости от координат квадранта на сфере
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

    def convert_index(self, lat0, lat1, lon0, lon1):
        """
        Переводим пару долгот и пару широт в индексы массивов.
        !!TODO: Сделать переход через линию перемены дат.
        :return:
        """
        idx_lat0 = int((1 + lat0 / 90) * 0.5 * self.lats)
        idx_lat1 = int((1 + lat1 / 90) * 0.5 * self.lats)
        idx_lon0 = int((1 + lon0 / 180) * 0.5 * self.lons)
        idx_lon1 = int((1 + lon1 / 180) * 0.5 * self.lons)
        return idx_lat0, idx_lat1, idx_lon0, idx_lon1

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
        idx_lat0, idx_lat1, idx_lon0, idx_lon1 = self.convert_index(lat0, lat1, lon0, lon1)
        return np.linalg.multi_dot([binary_image[idx_lat0:idx_lat1, idx_lon0:idx_lon1].flatten(),
                                    self.area_grid[idx_lat0:idx_lat1, idx_lon0:idx_lon1].flatten()])

    def integrate_field(self, field):
        """
        Интеграл по изображению, спроецированному на сферу
        :param field: Изображение
        :return:
        """
        return np.linalg.multi_dot([field.flatten(), self.area_grid.flatten()])

    def integrate_field_quad(self, field, lat0, lat1, lon0, lon1):
        """
        Интеграл по изображению, спроецированному на сферу
        в заданном квадранте сферы
        :param field: Изображение
        :return:
        """
        idx_lat0, idx_lat1, idx_lon0, idx_lon1 = self.convert_index(lat0, lat1, lon0, lon1)
        return np.linalg.multi_dot([field[idx_lat0:idx_lat1, idx_lon0:idx_lon1].flatten(),
                                    self.area_grid[idx_lat0:idx_lat1, idx_lon0:idx_lon1].flatten()])

    @staticmethod
    def sphere_arc_distance(lon1, lat1, lon2, lat2):
        """
        Угловая величина дуги, соединяющей две точки на сфере.
        """
        # See "Haversine formula"
        lon1, lat1, lon2, lat2 = np.deg2rad(lon1), np.deg2rad(lat1), np.deg2rad(lon2), np.deg2rad(lat2)
        a = np.sin((lat2 - lat1) / 2.0) ** 2
        b = np.sin((lon2 - lon1) / 2.0) ** 2
        c = np.sqrt(a + np.cos(lat2) * np.cos(lat1) * b)
        return 2 * np.arcsin(c)

    def sphere_distance(self, lon1, lat1, lon2, lat2):
        """
        Длина дуги, соединяющей две точки на сфере.
        """
        return self.sphere_arc_distance(lon1, lat1, lon2, lat2) * self.r

    def sphere_contour_distance(self, contour):
        """
        Длина контура на сфере.
        :param contour: последовательность точек (lon, lat).
        :return: длина контура
        """
        return np.sum([self.sphere_distance(contour[i][0], contour[i][1], contour[i + 1][0], contour[i + 1][1])
                       for i in range(len(contour) - 1)])

    def sphere_angle(self, lon1, lat1, lon2, lat2, lon3, lat3):
        """
        Угол ABC,
        где точки A(x1, y1), B(x2, y2), C(x3, y3)
        заданы в сферических координатах.
        """
        a = self.sphere_arc_distance(lon1, lat1, lon2, lat2)
        b = self.sphere_arc_distance(lon2, lat2, lon3, lat3)
        c = self.sphere_arc_distance(lon3, lat3, lon1, lat1)
        angle = np.arccos((np.cos(c) - np.cos(a) * np.cos(b)) / (np.sin(a) * np.sin(b)))
        return angle

    def _convert_to_spherical(self, contour):
        sph_x = [self.lonslist[int(p[0])] for p in contour]
        sph_y = [self.latslist[int(p[1])] for p in contour]
        return np.transpose([sph_x, sph_y])


def _test():
    import matplotlib.pyplot as plt

    s = SphericalGeometry(1, 1440, 720)
    print(s.area_grid.sum())
    plt.matshow(s.area_grid)
    plt.colorbar()
    plt.show()


def _test2():
    s = SphericalGeometry(1, 1440, 720)
    im = np.ones((720, 1440)) * 10
    # im[:, :720] = 0
    print(s.integrate_field_quad(im, 0, 90, -180, 180))
    print(s.integrate_field(im))


def _test3():
    lat1, lon1 = 55.5807418, 36.8237457  # Москва
    lat2, lon2 = 57.6521473, 39.583654   # Ярославль
    lat3, lon3 = 57.7696034, 40.8032077  # Кострома
    s = SphericalGeometry(6400, 1440, 720)
    print(s.sphere_distance(lon1, lat1, lon2, lat2))
    print(s.sphere_contour_distance([(lon1, lat1), (lon2, lat2), (lon3, lat3)]))
