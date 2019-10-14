import numpy as np
import morse.field_generator as gen
import skimage.filters
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plot3d
import threading
import time


def test():
    # Возьмём исходное изображение на слое z = 0
    image = gen.gen_field_from_file(r"C:\repo\pproc\data\input.fits", conditions="plain", filetype="fits")
    image = image[100:200]
    j = ScaleSpaceMesh3D(image, level_num=50)
    j.calc_scale_space_set(start_level=6)
    # j.draw()
    # plt.show()


class ScaleSpaceMesh3D:
    """
    Scale-space graph computation via Jacobi sets.
    References:
    Jacobi sets (discrete computation):
    Jacobi Sets of Multiple Morse Functions. // H. Edelsbrunner, J. Harer.
    Scale Space:
    Lindeberg T. // Scale-Space Theory in Computer Vision, 1994.
    Tracking of critical points via Jacobi sets:
    Bremer, P.-T., Bringa, E. M., Duchaineau, M. A., Gyulassy, A.,
    Laney, D., Mascarenhas, A., Pascucci, V., 2007.
    Topological feature extraction and tracking. Journal of Physics: Conference Series 78.
    """
    EDGE_TYPES = ['Xx', 'Yy', 'Zz', 'Xp', 'Xm', 'Yp', 'Ym', 'Zp', 'Zm']

    # Каждый нижний линк кодируется последовательностью 0 и 1: 0, если вершина нне присутствует,
    # 1 - если присутствует. Далее, каждую строчку представляем двоичным числом и переводим в десятичную.
    # Для 4-линков есть 16 типов линков, для 6-линков - 24.
    # Номеру линка сопоставляем его топологический тип: 0, если он стягиваем в точку,
    # 1 - если он гомеоморфен сфере (какой-нибудь степени), 2 - если это линки вида 101010 или 010101.
    LINK_4_TYPES = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    LINK_6_TYPES = [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0,
                    0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0,
                    0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 0,
                    0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    LINK_TYPES = dict([('0000', 1), ('0001', 0), ('0010', 0), ('0011', 0),
                       ('0100', 0), ('0101', 1), ('0110', 0), ('0111', 0),
                       ('1000', 0), ('1001', 0), ('1010', 1), ('1011', 0),
                       ('1100', 0), ('1101', 0), ('1110', 0), ('1111', 1),
                       ('000000', 1), ('000001', 0), ('000010', 0), ('000011', 0),
                       ('000100', 0), ('000101', 1), ('000110', 0), ('000111', 0),
                       ('001000', 0), ('001001', 1), ('001010', 1), ('001011', 1),
                       ('001100', 0), ('001101', 1), ('001110', 0), ('001111', 0),
                       ('010000', 0), ('010001', 1), ('010010', 1), ('010011', 1),
                       ('010100', 1), ('010101', 2), ('010110', 1), ('010111', 1),
                       ('011000', 0), ('011001', 1), ('011010', 1), ('011011', 1),
                       ('011100', 0), ('011101', 1), ('011110', 0), ('011111', 0),
                       ('100000', 0), ('100001', 0), ('100010', 1), ('100011', 0),
                       ('100100', 1), ('100101', 1), ('100110', 1), ('100111', 0),
                       ('101000', 1), ('101001', 1), ('101010', 2), ('101011', 1),
                       ('101100', 1), ('101101', 1), ('101110', 1), ('101111', 0),
                       ('110000', 0), ('110001', 0), ('110010', 1), ('110011', 0),
                       ('110100', 1), ('110101', 1), ('110110', 1), ('110111', 0),
                       ('111000', 0), ('111001', 0), ('111010', 1), ('111011', 0),
                       ('111100', 0), ('111101', 0), ('111110', 0), ('111111', 1)])

    def __init__(self, data, level_num):
        self.sizeX, self.sizeY = data.shape

        # Количество уровней
        self.sizeZ = level_num
        self.f = np.zeros((self.sizeX, self.sizeY, self.sizeZ))
        self.f[:, :, 0] = data

        # Данные на уровне определяются свёрткой с Гауссовским ядром
        for level in range(1, level_num):
            self.f[:, :, level] = skimage.filters.gaussian(data, sigma=level)
        self.g = np.ones((self.sizeX, self.sizeY, self.sizeZ))
        for i in range(self.sizeZ):
            self.g[:, :, i] = i
        self.jacobi_set = dict()

    def _link_Zm(self, x, y, z):
        # Линк ребра вида z-минус, т. е. параллельного направлению (-1, 1, 0)
        return [(x, y + 1, z + 1), (x - 1, y, z + 1), (x - 1, y, z),
                (x - 1, y, z - 1), (x, y + 1, z - 1), (x, y + 1, z)]

    def _link_Zp(self, x, y, z):
        # Линк ребра вида z-плюс, т. е. параллельного направлению (1, 1, 0)
        return [(x + 1, y, z + 1), (x, y + 1, z + 1), (x, y + 1, z),
                (x, y + 1, z - 1), (x + 1, y, z - 1), (x + 1, y, z)]

    def _link_Xm(self, x, y, z):
        # Линк ребра вида x-минус, т. е. параллельного направлению (0, 1, -1)
        return [(x + 1, y + 1, z), (x + 1, y, z - 1), (x, y, z - 1),
                (x - 1, y, z - 1), (x - 1, y + 1, z), (x, y + 1, z)]

    def _link_Xp(self, x, y, z):
        # Линк ребра вида x-плюс, т. е. параллельного направлению (0, 1, 1)
        return [(x + 1, y, z + 1), (x + 1, y + 1, z), (x, y + 1, z),
                (x - 1, y + 1, z), (x - 1, y, z + 1), (x, y, z + 1)]

    def _link_Ym(self, x, y, z):
        # Линк ребра вида y-минус, т. е. параллельного направлению (-1, 0, 1)
        return [(x, y + 1, z + 1), (x - 1, y + 1, z), (x - 1, y, z),
                (x - 1, y - 1, z), (x, y - 1, z + 1), (x, y, z + 1)]

    def _link_Yp(self, x, y, z):
        # Линк ребра вида y-плюс, т. е. параллельного направлению (1, 0, 1)
        return [(x + 1, y + 1, z), (x, y + 1, z + 1), (x, y, z + 1),
                (x, y - 1, z + 1), (x + 1, y - 1, z), (x + 1, y, z)]

    def _link_Xx(self, x, y, z):
        # Линк ребра вида x-x, т. е. параллельного направлению (1, 0, 0)
        # Квадрат строится вокруг белой вершины.
        if (x + y + z) % 2 == 1:
            x = x + 1
        return [(x, y + 1, z), (x, y, z - 1), (x, y - 1, z), (x, y, z + 1)]

    def _link_Yy(self, x, y, z):
        # Линк ребра вида y-y, т. е. параллельного направлению (0, 1, 0)
        # Квадрат строится вокруг белой вершины.
        if (x + y + z) % 2 == 1:
            y = y + 1
        return [(x, y, z + 1), (x - 1, y, z), (x, y, z - 1), (x + 1, y, z)]

    def _link_Zz(self, x, y, z):
        # Линк ребра вида z-z, т. е. параллельного направлению (0, 0, 1)
        # Квадрат строится вокруг белой вершины.
        if (x + y + z) % 2 == 1:
            z = z + 1
        return [(x + 1, y, z), (x, y - 1, z), (x - 1, y, z), (x, y + 1, z)]

    def _get_link(self, x, y, z, edge_type):
        """
        Линк ребра по начальной точке и типу ребра.
        :param x: X coordinate
        :param y: Y coordinate
        :param z: Z coordinate
        :param edge_type: edge type from EDGE_TYPE list.
        :return:
        """
        link = []
        link_size = 0
        if edge_type == 'Xx':
            link = self._link_Xx(x, y, z)
            link_size = 4
        elif edge_type == 'Yy':
            link = self._link_Yy(x, y, z)
            link_size = 4
        elif edge_type == 'Zz':
            link = self._link_Zz(x, y, z)
            link_size = 4
        elif edge_type == 'Xm':
            link = self._link_Xm(x, y, z)
            link_size = 6
        elif edge_type == 'Xp':
            link = self._link_Xp(x, y, z)
            link_size = 6
        elif edge_type == 'Ym':
            link = self._link_Ym(x, y, z)
            link_size = 6
        elif edge_type == 'Yp':
            link = self._link_Yp(x, y, z)
            link_size = 6
        elif edge_type == 'Zm':
            link = self._link_Zm(x, y, z)
            link_size = 6
        elif edge_type == 'Zp':
            link = self._link_Zp(x, y, z)
            link_size = 6
        return link, link_size

    def _get_edge_end(self, x, y, z, edge_type):
        """
        Координаты конца ребра по его типу и начальной точке.
        :param x: X coordinate
        :param y: Y coordinate
        :param z: Z coordinate
        :param edge_type: edge type from EDGE_TYPE list.
        :return:
        """
        if edge_type == 'Xx':
            return x + 1, y, z
        elif edge_type == 'Yy':
            return x, y + 1, z
        elif edge_type == 'Zz':
            return x, y, z + 1
        elif edge_type == 'Xm':
            return x, y + 1, z - 1
        elif edge_type == 'Xp':
            return x, y + 1, z + 1
        elif edge_type == 'Ym':
            return x - 1, y, z + 1
        elif edge_type == 'Yp':
            return x + 1, y, z + 1
        elif edge_type == 'Zm':
            return x - 1, y + 1, z
        elif edge_type == 'Zp':
            return x + 1, y + 1, z
        else:
            return None

    def _get_edge_criticality(self, x, y, z, edge_type):
        """
        Считаем порядок вхождения критического ребра.
        Если у нижнего линка один разрыв - то ребро не критическое,
        если 0 или 2 - то критическое порядка 1,
        если 3 - то порядка 2 (паттерн 010101).
        :param x: X coordinate.
        :param y: Y coordinate.
        :param z: Z coordinate.
        :param edge_type: edge type from EDGE_TYPE list.
        :return:
        """
        x1, y1, z1 = self._get_edge_end(x, y, z, edge_type)  # Находим конец ребра.

        # Сначала найдём множитель l = \lambda, при котором h(x) = g(x) + l f(x) на концах ребра равна.
        # (функция g может принимать одно значение, поэтому загоняем разность g в числитель,
        #  а функцию f мы шевелим, поэтому её отправляем в знаменатель)
        dif_g = self.g[x1, y1, z1] - self.g[x, y, z]
        dif_f = self.f[x1, y1, z1] - self.f[x, y, z]
        k = -dif_g / dif_f

        # Значение функции h при найденном l = \lambda
        h = self.g[x, y, z] + k * self.f[x, y, z]
        link, link_size = self._get_link(x, y, z, edge_type)

        # Кодируем нижний линк двоичным кодом (1 - если точка присутствует, 2 - если нет).
        # Переводим в десятичный и смотрим тип линка по спискам LINK_4_TYPES и LINK_6_TYPES.
        lower_link_type = 0
        two_power = 1
        for i in range(link_size):
            v = link[i]
            if self.g[v[0], v[1], v[2]] + k * self.f[v[0], v[1], v[2]] < h:
                lower_link_type += two_power
            two_power *= 2

        return self.LINK_4_TYPES[lower_link_type] if link_size == 4 else self.LINK_6_TYPES[lower_link_type]

    def calc_scale_space_set(self, start_level=3):
        """
        Считаем множество Якоби, не включая граничные рёбра (чтобы не усложнять вычисление линка)
        :param start_level: Уровень, начиная с которого вычисляется множество Якоби (минимум - 0).
        :return:
        """
        self.jacobi_set = dict()
        for edge_type in ScaleSpaceMesh3D.EDGE_TYPES:
            self.jacobi_set[edge_type] = []

        for i in range(1, self.sizeX - 1):
            print('\u25A0', end='')
            for j in range(1, self.sizeY - 1):
                for k in range(start_level, self.sizeZ - 1):
                    # Прямые рёбра выходят из всех вершин.
                    # Горизонтальные рёбра в множество не входят.
                    for edge_type in ['Zz', 'Xx', 'Yy']:
                        if self._get_edge_criticality(i, j, k, edge_type) > 0:
                            self.jacobi_set[edge_type].append([(i, j, k), self._get_edge_end(i, j, k, edge_type)])
                    # Диагональные рёбра выходят только из чёрных вершин.
                    if (i + j + k) % 2 == 1:
                        for edge_type in ['Xp', 'Xm', 'Yp', 'Ym', 'Zp', 'Zm']:
                            if self._get_edge_criticality(i, j, k, edge_type) > 0:
                                self.jacobi_set[edge_type].append([(i, j, k), self._get_edge_end(i, j, k, edge_type)])

    def calc_scale_space_set_parallel(self, start_level=3):
        """
        Считаем множество Якоби параллельно.
        Вычисляем только рёбра видов Zz, Xp, Xm, Yp, Ym,
        поскольку других при нашем определении функций быть не может.

        Пока распараллеливание даёт выигрыш в 2 - 2.5 раза.
        Надо улучшать, попробовать параллелить не по типу рёбер на большее число потоков.
        :param start_level: Уровень, с которого начинаем вычисления.
        :return: в поле jacobi_set записывается множество Якоби.
        """

        self.jacobi_set = dict()
        for edge_type in ScaleSpaceMesh3D.EDGE_TYPES:
            self.jacobi_set[edge_type] = []

        # Мы будем писать в разные массивы рёбер в разных потоках.
        def _cmp_edge_set_straight(e_type):
            """
            Вычисление мнлжества Якоби из диагональных рёбер заданного типа.
            :param e_type:
            :return:
            """
            result = []
            for i in range(1, self.sizeX - 1):
                for j in range(1, self.sizeY - 1):
                    for k in range(start_level, self.sizeZ - 1):
                        if self._get_edge_criticality(i, j, k, e_type) > 0:
                            result.append([(i, j, k), self._get_edge_end(i, j, k, e_type)])
            self.jacobi_set[e_type] = result

        def _cmp_edge_set_diagonal(e_type):
            """
            Вычисление мнлжества Якоби из диагональных рёбер заданного типа.
            :param e_type:
            :return:
            """
            result = []
            for i in range(1, self.sizeX - 1):
                for j in range(1, self.sizeY - 1):
                    for k in range(start_level, self.sizeZ - 1):
                        # Диагональные рёбра выходят только из чёрных вершин.
                        if (i + j + k) % 2 == 1:
                            if self._get_edge_criticality(i, j, k, e_type) > 0:
                                result.append([(i, j, k), self._get_edge_end(i, j, k, e_type)])
            self.jacobi_set[e_type] = result

        threads = []

        for edge_type in ['Xp', 'Xm', 'Yp', 'Ym']:
            my_thread = threading.Thread(target=_cmp_edge_set_diagonal, args=(edge_type,))
            threads.append(my_thread)

        for edge_type in ['Zz']:
            my_thread = threading.Thread(target=_cmp_edge_set_straight, args=(edge_type,))
            threads.append(my_thread)

        for thr in threads:
            thr.start()

        # Method should finish after all threads are finished.
        for thr in threads:
            thr.join()

    def all_edges(self):
        """
        Список всех критических рёбер.
        :return:
        """
        pass

    def all_nodes(self):
        """
        Список всех вершин, из которых выходит критическое ребро
        :return:
        """
        pass

    @staticmethod
    def _draw_edge_set(ax, edge_set):
        """
        Рисуем набор рёбер на заданных 3D-осях.
        :param ax:
        :param edge_set:
        :return:
        """
        for e in edge_set:
            ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], [e[0][2], e[1][2]], '-k')
        return ax

    def draw(self):
        xx, yy = np.meshgrid(np.linspace(0, self.sizeX, self.sizeX), np.linspace(0, self.sizeY, self.sizeY))
        zz = np.zeros((self.sizeX, self.sizeY))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for edge_type in self.EDGE_TYPES:
            self._draw_edge_set(ax, self.jacobi_set[edge_type])


if __name__ == '__main__':
# if False:
    import argparse
    import networkx as nx
    import scipy.interpolate as sip
    # Эта штука нужна, чтобы рисовать в 3D с картинкой на фоне.
    import visvis as vv
    from visvis.utils.pypoints import Pointset
    vv.use('qt5')

    print("Started...")

    parser = argparse.ArgumentParser(description='Scale space computation')
    parser.add_argument('--input', metavar='i', type=str, help='input FITS file')
    parser.add_argument('--output', metavar='o', type=str, help='output PNG file')
    parser.add_argument('--levels', metavar='l', type=int, default=50, help='number of levels', required=False)
    parser.add_argument('--parallel', metavar='p', type=int, default=True, help='parallel computation', required=False)
    args = parser.parse_args()

    image = gen.gen_field_from_file(args.input, conditions='plain', filetype='fits')
    j = ScaleSpaceMesh3D(image, level_num=args.levels)
    if args.parallel:
        print('Parallel computation started...')
        j.calc_scale_space_set_parallel(start_level=6)
    else:
        print('Computation started...')
        j.calc_scale_space_set(start_level=6)

    print("Jacobi graph constructing...")
    edges, nodes = [], []
    for edge_type in j.EDGE_TYPES:
        edges += j.jacobi_set[edge_type]

    for edge in edges:
        nodes += [edge[0], edge[1]]
    nodes = list(set(nodes))

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    print("Remove loops...")
    while True:
        try:
            g.remove_edges_from(nx.find_cycle(g))
        except nx.NetworkXNoCycle:
            break
    print("Remove isolated vertices...")
    g.remove_nodes_from([v for v in nodes if g.degree[v] == 0])

    print("Recompile into paths...")
    paths = list(nx.connected_components(g))
    print("Number of paths: {0}".format(len(paths)))

    def reduce_path_hops(p):
        # Удаляем из пути зигзаги (см. стр. 5 части Scale-Space диссера)
        i = 0
        while i < len(p) - 2:
            if p[i][2] == p[i + 2][2]:  # Эта вершина и вершина через одну лежат на одном уровне
                del p[i + 1]  # Удаляем промежуточную точку
            else:
                i += 1
    straight_paths = []
    for p in paths:
        p_endpoints = [node for node in p if g.degree[node] == 1]
        p_straight = nx.shortest_path(g, p_endpoints[0], p_endpoints[1])
        reduce_path_hops(p_straight)
        straight_paths.append(p_straight)

    print("Number of straight paths: {0}".format(len(straight_paths)))
    print("Redirecting paths...")
    for i in range(len(straight_paths)):
        if straight_paths[i][0][2] > straight_paths[i][-1][2]:
            straight_paths[i].reverse()

    sizeX, sizeY = image.shape[:2]

    def is_path_border(path):
        top_point = path[-1]
        return (top_point[0] == 0) or (top_point[0] == 0) or (top_point[0] == sizeX - 1) or (
                    top_point[0] == sizeX - 2) or \
               (top_point[1] == 0) or (top_point[1] == 1) or (top_point[1] == sizeY - 1) or (top_point[1] == sizeY - 2)

    print("Remove paths leading into boundary")
    normal_paths = [p for p in straight_paths if not is_path_border(p)]

    print("Number of correct paths: {0}".format(len(normal_paths)))
    long_normal_paths = [n for n in normal_paths if len(n) > 3]
    print("Number of long paths: {0}".format(len(long_normal_paths)))

    def path_top_point(path):
        """
        Ищем топ-точку пути.
        Находим все точки с максимальной координатой Z, берём среднее.
        """
        p = np.array(path, dtype=np.int)
        idx = np.where(p[:, 2] == p[:, 2].max())[0].mean(dtype=np.int)
        return path[idx]

    def draw_smoothed_path(ax, p, radius=1, color=(1, 0, 0), draw_source=False):
        if draw_source:
            x, y, z = np.transpose(p)
            vv.plot(x, y, z, '--r', axes=ax)
        tck, u = sip.splprep(np.transpose(p), k=3)
        new_p = sip.splev(np.linspace(0, 1, 200), tck)

        pp = Pointset(3)
        for pt in np.transpose(new_p):
            pp.append((pt[0], pt[1], pt[2]))

        #     line = vv.solidLine(pp, radius, 8, axes=ax)
        #     line = vv.line(pp, radius, 8, axes=ax)
        vv.plot(pp, lw=3, lc=color, alpha=0.5, ls='-')

    def draw_top_points(ax, paths, color=(1, 1, 0)):
        top_pts = np.array([path_top_point(p) for p in paths])
        pp = Pointset(top_pts)
        vv.plot(pp, ms='.', mc=color, mw='30', ls='', mew=0, axes=ax)

    def plot_bg_image(imgdata, level):
        nr, nc = imgdata.shape[:2]
        x, y = np.mgrid[:nr, :nc]
        z = np.ones((nr, nc))
        vv.functions.surf(x, y, z * level, imgdata, aa=3)

    # Рисуем картинки
    z_multiplier = 4  # Увеличиваем масштаб по Z
    plot_bg_image(image, level=6)
    ax = vv.gca()
    ax.bgcolor = (0.7, 0.7, 0.7)
    ax.light0.position = (200, 200, 200, 0)
    for path in long_normal_paths:
        if len(path) <= 3:
            # Для кубического сплайна нужно 4 точки.
            continue
        draw_smoothed_path(ax, path, radius=2, color=(1, 1, 1))
    draw_top_points(ax, long_normal_paths)
    ax.camera.azimuth = 45.
    ax.camera.elevation = 20.
    ax.camera.daspect = (1, 1, z_multiplier)
    vv.gcf().relativeFontSize = 2.0

    # import time
    # print("Wait for rendering...")
    # time.sleep(30)
    # vv.screenshot(args.output, ob=ax)
