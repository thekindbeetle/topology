import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from morse.unionfind import UnionFind as UF


class ReebGraph:

    def __init__(self, data):
        self.data = data  # Исходные данные
        self.shape = data.shape
        self.index = None  # Ячейки в порядке возрастания значений
        self.cell_num = data.size
        self.order = np.zeros(data.shape)  # Номера ячеек на сетке

        self.critical_index = []
        self.colors = []

        self.merge_tree = []
        self.split_tree = []

        self.split_graph = nx.DiGraph()
        self.merge_graph = nx.DiGraph()
        self.reeb_graph_augmented = nx.DiGraph()
        self.reeb_graph_contracted = nx.DiGraph()


    @staticmethod
    def _reduce_node(graph, node):
        """
        Remove node from graph.
        Works only with 1-1 degree and 0-1 degree vertices.
        Connect neighbour vertices.
        Remove flag shows if we need to delete not 1-1 node.
        """
        if graph.in_degree(node) == 1 and graph.out_degree(node) == 1:
            prev_neighbour = list(graph.predecessors(node))[0]
            next_neighbour = list(graph.successors(node))[0]
            graph.add_edge(prev_neighbour, next_neighbour)
        graph.remove_node(node)

    def reduce_node_and_set_persistence(self, node):
        """
        Reduce node from graph if it is possible.
        Set persistence to the new edge
        :param node:
        :return:
        """
        graph = self.reeb_graph_contracted
        if graph.in_degree(node) == 1 and graph.out_degree(node) == 1:
            prev_neighbour = list(graph.predecessors(node))[0]
            next_neighbour = list(graph.successors(node))[0]
            graph.add_edge(prev_neighbour, next_neighbour,
                           persistence=self._cmp_persistence_for_edge(prev_neighbour, next_neighbour))
            graph.remove_node(node)

    def _reduce_edge(self, e):
        """
        Contract edge in graph.
        We allow only saddle-extrema edge contraction.
        We check that after removing of extremum there are one downward and one upward (at least) direction.
        """
        graph = self.reeb_graph_contracted
        if graph.node[e[0]]['morse_index'] == 0:
            print('Minimum ', e[0])
            print('Saddle ', e[1])

            if graph.out_degree(e[0]) == 1 and graph.in_degree(e[1]) >= 2:
                graph.remove_node(e[0])
                print('Remove', e[0])
                return True
            else:
                return False
            # self.reduce_node_and_set_persistence(e[1])
            # print('Reduce', e[1])
        elif graph.node[e[1]]['morse_index'] == 2:
            print('Maximum ', e[1])
            print('Saddle ', e[0])

            if graph.in_degree(e[1]) == 1 and graph.out_degree(e[0]) >= 2:
                graph.remove_node(e[1])
                print('Remove', e[1])
                return True
            else:
                return False
            # self.reduce_node_and_set_persistence(e[0])
            # print('Reduce', e[0])
        else:
            return False

    def _cmp_persistence_for_edge(self, v0, v1):
        return self.data[self.index[v1][0], self.index[v1][1]] - \
               self.data[self.index[v0][0], self.index[v0][1]]

    def _neighbours(self, i, j):
        if i == 0:
            if j == 0:
                return [(0, 1), (1, 0)]
            elif j == self.shape[1] - 1:
                return [(0, j - 1), (1, j)]
            else:
                return [(0, j - 1), (1, j), (0, j + 1)]
        elif i == self.shape[0] - 1:
            if j == 0:
                return [(i, 1), (i - 1, 0)]
            elif j == self.shape[1] - 1:
                return [(i, j - 1), (i - 1, j)]
            else:
                return [(i, j - 1), (i - 1, j), (i, j + 1)]
        else:
            if j == 0:
                return [(i - 1, j), (i, j + 1), (i + 1, j)]
            elif j == self.shape[1] - 1:
                return [(i - 1, j), (i, j - 1), (i + 1, j)]
            else:
                return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

    def make_index(self):
        """
        Sort cells by its values.
        :return: List of pairs [x, y] of array cells sorted by values.
        """
        y, x = np.meshgrid(range(self.shape[1]), range(self.shape[0]))
        flatten_data = self.data.flatten()
        self.index = np.dstack((x.flatten()[flatten_data.argsort()],
                                y.flatten()[flatten_data.argsort()]))[0]
        self.order = np.zeros(self.cell_num, dtype=np.int)
        self.order[flatten_data.argsort()] = np.arange(0, self.cell_num, dtype=np.int)
        self.order = self.order.reshape(self.shape)

    def cmp_merge_and_split_trees(self):
        """
        See Carr, H., Snoeyink, J., & Axen, U. (2003).
        Computing contour trees in all dimensions.
        Computational Geometry, 24(2), 75–94.
        """
        # Compute merge tree
        uf = UF(self.cell_num)
        highest_vertex = dict()
        for i in range(self.cell_num):
            uf.makeset(i)
            highest_vertex[i] = i
            for neighbour in self._neighbours(*self.index[i]):
                j = self.order[neighbour[0], neighbour[1]]
                if (j > i) or (uf.find(j) == uf.find(i)):
                    pass
                else:
                    self.merge_tree.append([highest_vertex[uf.find(j)], i])
                    uf.union(i, j)
                    highest_vertex[uf.find(j)] = i

        # Compute split tree
        uf = UF(self.cell_num)
        lowest_vertex = dict()
        for i in reversed(range(self.cell_num)):
            uf.makeset(i)
            lowest_vertex[i] = i
            for neighbour in self._neighbours(*self.index[i]):
                j = self.order[neighbour[0], neighbour[1]]
                if (j < i) or (uf.find(j) == uf.find(i)):
                    pass
                else:
                    self.split_tree.append([i, lowest_vertex[uf.find(j)]])
                    uf.union(i, j)
                    lowest_vertex[uf.find(j)] = i

    def convert_to_nx_graphs(self):
        self.split_graph = nx.DiGraph()
        self.merge_graph = nx.DiGraph()
        self.split_graph.add_nodes_from(np.arange(0, self.cell_num, dtype=np.int))
        self.merge_graph.add_nodes_from(np.arange(0, self.cell_num, dtype=np.int))
        self.split_graph.add_edges_from(self.split_tree)
        self.merge_graph.add_edges_from(self.merge_tree)

    def set_critical_index_and_colors(self):
        self.critical_index = [-1 for i in range(self.cell_num)]
        self.colors = [(0, 0, 0) for i in range(self.cell_num)]
        for i in range(self.cell_num):
            if (self.reeb_graph_augmented.in_degree(i) == 0): # and (self.reeb_graph_augmented.out_degree(i) == 1):
                self.critical_index[i] = 0
                self.colors[i] = (0, 0, 1)
            elif (self.reeb_graph_augmented.out_degree(i) == 0): # and (self.reeb_graph_augmented.out_degree(i) == 0):
                self.critical_index[i] = 2
                self.colors[i] = (1, 0, 0)
            else: # (self.reeb_graph_augmented.in_degree(i) > 1) or (self.reeb_graph_augmented.out_degree(i) > 1):
                self.critical_index[i] = 1
                self.colors[i] = (0, 1, 0)
        colors_dict = dict([(v, self.colors[v]) for v in range(self.cell_num)])
        critical_index_dict = dict([(v, self.critical_index[v]) for v in range(self.cell_num)])
        nx.set_node_attributes(self.reeb_graph_augmented, colors_dict, 'color')
        nx.set_node_attributes(self.reeb_graph_augmented, critical_index_dict, 'morse_index')

    def cmp_augmented_reeb_graph(self):
        """
        See Carr, H., Snoeyink, J., & Axen, U. (2003).
        Computing contour trees in all dimensions.
        Computational Geometry, 24(2), 75–94.
        """
        # 1. For each vertex xi, if up-degree in join tree + down-degree in split tree is 1, enqueue x_i.
        leaf_queue = deque(reversed([x for x in range(self.cell_num) if
                                     self.merge_graph.in_degree(x) + self.split_graph.out_degree(x) == 1]))

        # 2. Initialize contour tree to an empty graph on ||join tree|| vertices.
        self.reeb_graph_augmented = nx.DiGraph()
        self.reeb_graph_augmented.add_nodes_from(range(self.cell_num))

        # 3. While leaf queue size > 1
        while leaf_queue:
            # Dequeue the first vertex, xi, on the leaf queue.
            x = leaf_queue.pop()

            # If xi is an upper leaf
            if self.merge_graph.in_degree[x] == 0:
                if self.split_graph.out_degree[x] == 0:
                    break  # Вершина без рёбер

                # find incident arc y_i y_j in join tree.
                neighbour = list(self.merge_graph.successors(x))[0]
                self.reeb_graph_augmented.add_edge(x, neighbour)
            else:
                # else find incident arc z_iz_j in split tree.
                neighbour = list(self.split_graph.predecessors(x))[0]
                self.reeb_graph_augmented.add_edge(neighbour, x)

            # Remove node from split and merge graphs
            # See Definition 4.5 from H. Carr (2003)
            self._reduce_node(self.merge_graph, x)
            self._reduce_node(self.split_graph, x)

            # If x_j is now a leaf, enqueue x_j
            if self.merge_graph.in_degree(neighbour) + self.split_graph.out_degree(neighbour) == 1:
                leaf_queue.append(neighbour)

    def contract_edges(self):
        """
        Contract edges of the augmented Reeb graph.
        Retain only critical points.
        """
        self.reeb_graph_contracted = self.reeb_graph_augmented.copy(as_view=False)
        nodes_to_remove = [node for node in self.reeb_graph_contracted.nodes
                           if self.reeb_graph_contracted.in_degree(node) == 1 and
                              self.reeb_graph_contracted.out_degree(node) == 1]
        for node in nodes_to_remove:
            self._reduce_node(self.reeb_graph_contracted, node)

    def set_persistence(self):
        persistence = dict([(e, self._cmp_persistence_for_edge(e[0], e[1])) for e in self.reeb_graph_contracted.edges])
        nx.set_edge_attributes(self.reeb_graph_contracted, persistence, 'persistence')

    def draw(self):
        plt.matshow(np.transpose(self.data), cmap='gray')
        colors = list(nx.get_node_attributes(self.reeb_graph_contracted, 'color').values())
        positions = dict([(idx, self.index[idx]) for idx in range(self.cell_num)])
        nx.draw_networkx(self.reeb_graph_contracted, pos=positions, with_labels=False, node_size=50, node_color=colors)
        plt.colorbar()

    def draw_3d(self, draw_edges=False, use_threshold=True, threshold=20, xmin=0, ymin=0, xmax=None, ymax=None,
                zmin=-3000, zmax=3000, title='', fname=None):
        """
        :param draw_edges: Рисовать рёбра
        :param use_threshold: Не показывать маленькие экстремумы.
        :param threshold: Пороговое значение для отрисовки экстремумов (по модулю).
        :return:
        """
        import mpl_toolkits.mplot3d as plot3d

        if xmax is None:
            xmax = self.data.shape[1]
        if ymax is None:
            ymax = self.data.shape[0]

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(range(self.data.shape[0]), range(self.data.shape[1]))
        ax.plot_wireframe(x, y, np.transpose(self.data), colors='gray', linewidths=0.5)
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.set_zlim((zmin, zmax))
        ax.set_title(title)

        if draw_edges:
            edges = []
            for e in self.reeb_graph_contracted.edges():
                ax.plot([self.index[e[0]][0], self.index[e[1]][0]], [self.index[e[0]][1], self.index[e[1]][1]],
                        [self.data[self.index[e[0]][0], self.index[e[0]][1]],
                         self.data[self.index[e[1]][0], self.index[e[1]][1]]], '-k')

        if use_threshold:
            nodes = [v for v in self.reeb_graph_contracted.nodes() if
                     np.abs(self.data[self.index[v][0], self.index[v][1]]) > threshold]

            colors_dict = nx.get_node_attributes(self.reeb_graph_contracted, 'color')
            colors = [colors_dict[v] for v in nodes]
            positions_x = [self.index[idx][0] for idx in nodes]
            positions_y = [self.index[idx][1] for idx in nodes]
            positions_z = [self.data[self.index[idx][0], self.index[idx][1]] for idx in nodes]
            ax.scatter(positions_x, positions_y, positions_z, c=colors, s=100)
        else:
            colors = list(nx.get_node_attributes(self.reeb_graph_contracted, 'color').values())
            positions_x = [self.index[idx][0] for idx in self.reeb_graph_contracted.nodes()]
            positions_y = [self.index[idx][1] for idx in self.reeb_graph_contracted.nodes()]
            positions_z = [self.data[self.index[idx][0], self.index[idx][1]] for idx in self.reeb_graph_contracted.nodes()]

            ax.scatter(positions_x, positions_y, positions_z, c=colors, s=100)
        if fname is not None:
            fig.savefig(fname)
            plt.close()

    @staticmethod
    def build_all(data):
        """
        Static constructor of the Reeb graph by 2D-data.
        :param data:
        :return:
        """
        reeb = ReebGraph(data)
        reeb.make_index()
        reeb.cmp_merge_and_split_trees()
        reeb.convert_to_nx_graphs()
        reeb.cmp_augmented_reeb_graph()
        reeb.set_critical_index_and_colors()
        reeb.contract_edges()
        reeb.set_persistence()
        return reeb


def test():
    import morse.field_generator
    # data = np.array([
    #     [1.0, 2.0, 1.5],
    #     [2.5, 5.0, 1.6],
    #     [1.2, 2.2, 1.7],
    #     [2.5, 5.0, 1.6],
    #     [1.0, 2.0, 1.5]
    # ])
    # data = morse.field_generator.\
    #     gen_gaussian_sum_rectangle(50, 50, [(10, 10), (15, 15), (10, 15), (20, 5)], 3)

    # data = morse.field_generator.gen_sincos_field(200, 200, 0.3, 0.2)
    # data = morse.field_generator.gen_field_from_file("C:/data/test.bmp", conditions='plain')
    # data = morse.field_generator.gen_field_from_file(
    #     "C:/data/hmi/processed/AR12673/hmi_m_45s_2017_09_06_07_24_45_tai_magnetogram.fits",
    #     filetype='fits',
    #     conditions='plain')
    #
    import os
    import skimage.filters

    datadir = "C:/data/hmi/processed/AR12673/"
    files = os.listdir(datadir)
    for file in files:
        data = morse.field_generator.gen_field_from_file(
            os.path.join(datadir, file),
            filetype='fits',
            conditions='plain')

        data = skimage.filters.gaussian(data, sigma=10)
        data = data[100:300, 100:300]

        reeb = ReebGraph(data)
        reeb.make_index()
        reeb.cmp_merge_and_split_trees()
        reeb.convert_to_nx_graphs()
        reeb.cmp_augmented_reeb_graph()
        reeb.set_critical_index_and_colors()
        reeb.contract_edges()
        reeb.set_persistence()

        # reeb.draw()
        print(nx.is_tree(reeb.reeb_graph_augmented))
        print(nx.is_tree(reeb.reeb_graph_contracted))

        persistence = nx.get_edge_attributes(reeb.reeb_graph_contracted, 'persistence')
        morse_index = nx.get_node_attributes(reeb.reeb_graph_contracted, 'morse_index')

        saddle_ext_arcs = [e for e in reeb.reeb_graph_contracted.edges if
                           ((morse_index[e[0]] == 0) or (morse_index[e[1]] == 2)) and persistence[e] < 20]

        for e in saddle_ext_arcs:
            try:
                reeb._reduce_edge(e)
            except:
                print('Fail')

        reeb.contract_edges()
        reeb.draw_3d(draw_edges=False, zmin=-2000, zmax=2000, title=file, fname='{0}.png'.format(file[:-5]))

test()