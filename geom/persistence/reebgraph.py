import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from morse.unionfind import UnionFind as UF


class ReebGraph:
    """
    Reeb Graph computation for data on quad mesh on the plane.
    """

    def __init__(self, data):
        self.data = data  # Initial data (2D-array)
        self.shape = data.shape
        self.index = None  # Ячейки в порядке возрастания значений
        self.cell_num = data.size
        self.order = np.zeros(data.shape)  # Номера ячеек на сетке

        self.merge_tree = []
        self.split_tree = []

        self.split_graph = nx.DiGraph()
        self.merge_graph = nx.DiGraph()
        self.reeb_graph_augmented = nx.DiGraph()
        self.reeb_graph_contracted = nx.DiGraph()

    def _reduce_node(self, graph, node, set_persistence=False):
        """
        Remove node from graph.
        Works only with 1-1 degree and 0-1 degree vertices.
        Connect neighbour vertices.
        Remove flag shows if we need to delete not 1-1 node.
        """
        if graph.in_degree(node) == 1 and graph.out_degree(node) == 1:
            prev_neighbour = list(graph.predecessors(node))[0]
            next_neighbour = list(graph.successors(node))[0]
            if set_persistence:
                graph.add_edge(prev_neighbour, next_neighbour,
                               persistence=graph.node[next_neighbour]['value'] - graph.node[prev_neighbour]['value'])
            else:
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
                           persistence=graph[next_neighbour]['value'] - graph[prev_neighbour]['value'])
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
            # Здесь проблема в том, что может быть добавлено сразу несколько рёбер.
            # Можно попробовать решить так: если добавляется более двух соседей, добавлять ещё одну вершину.
            # Но этого я пока делать не хочу.
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
            # for neighbour in self._neighbours(*self.index[i]):
            for neighbour in reversed(self._neighbours(*self.index[i])):
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
        morse_index = dict([(node, -1) for node in self.reeb_graph_augmented.nodes])
        colors = dict([(node, (0, 0, 0)) for node in self.reeb_graph_augmented.nodes])

        for i in self.reeb_graph_augmented.nodes:
            if self.reeb_graph_augmented.in_degree(i) == 0:# and (self.reeb_graph_augmented.out_degree(i) == 1):
                morse_index[i] = 0
                colors[i] = (0, 0, 1)
            elif self.reeb_graph_augmented.out_degree(i) == 0:# and (self.reeb_graph_augmented.in_degree(i) == 1):
                morse_index[i] = 2
                colors[i] = (1, 0, 0)
            else: # (self.reeb_graph_augmented.in_degree(i) > 1) or (self.reeb_graph_augmented.out_degree(i) > 1):
                morse_index[i] = 1
                colors[i] = (0, 0.5, 0)
        nx.set_node_attributes(self.reeb_graph_augmented, colors, 'color')
        nx.set_node_attributes(self.reeb_graph_augmented, morse_index, 'morse_index')

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
                
    def set_node_values_and_positions(self):
        """
        Set values and positions on nodes.
        """
        values = dict([(node, self.data[self.index[node][0], self.index[node][1]])
                        for node in range(self.cell_num)])       
        positions = dict([(node, [self.index[node][1], self.index[node][0]])
                        for node in range(self.cell_num)])
        nx.set_node_attributes(self.reeb_graph_augmented, values, name="value")
        nx.set_node_attributes(self.reeb_graph_augmented, positions, name="position")

    def check_graph_consistency(self):
        """
        Могут случаться вырождения (таки мы с картинкой работаем, а не с триангуляцией) такого плана.
        Возникает минимум, соединённый с двумя сёдлами, либо максимум с двумя сёдлами.
        Мы заменяем его на седло, а минимум прикрепляем к нему.
        :return:
        """
        g = self.reeb_graph_augmented
        for v in list(g.nodes):
            if g.in_degree(v) == 0 and g.out_degree(v) > 1:
                # Это минимум, который соединён с двумя вершинами.
                # Он будет седлом, а к нему мы присобачим минимум.
                new_node = v + 0.5
                g.add_node(new_node, value=g.node[v]['value'],
                                     position=(g.node[v]['position'][0] + 0.5, g.node[v]['position'][1] + 0.5))
                g.add_edge(new_node, v)
            if g.out_degree(v) == 0 and g.in_degree(v) > 1:
                # Это максимум, который соединён с двумя вершинами.
                # Он будет седлом, а к нему мы присобачим максимум.
                new_node = v + 0.5
                g.add_node(new_node, value=g.node[v]['value'],
                                     position=(g.node[v]['position'][0] + 0.5, g.node[v]['position'][1] + 0.5))
                g.add_edge(v, new_node)

    def contract_edges(self):
        """
        Contract edges of the augmented Reeb graph.
        Retain only critical points.
        """
        nodes_to_remove = [node for node in self.reeb_graph_contracted.nodes
                           if self.reeb_graph_contracted.in_degree(node) == 1 and
                              self.reeb_graph_contracted.out_degree(node) == 1]
        for node in nodes_to_remove:
            self._reduce_node(self.reeb_graph_contracted, node, set_persistence=True)

    def set_persistence(self):
        persistence = dict([(e, self.reeb_graph_contracted.node[e[1]]['value'] -
                                self.reeb_graph_contracted.node[e[0]]['value'])
                            for e in self.reeb_graph_contracted.edges])
        nx.set_edge_attributes(self.reeb_graph_contracted, persistence, 'persistence')

    def draw(self, background=None, vmin=-2000, vmax=2000):
        """
        :param background: background image (By default, self.data)
        """
        if background is None:
            background = self.data
        plt.matshow(background, cmap='gray', vmin=vmin, vmax=vmax)
        colors = list(nx.get_node_attributes(self.reeb_graph_contracted, 'color').values())
        positions = nx.get_node_attributes(self.reeb_graph_contracted, 'position')
        nx.draw_networkx(self.reeb_graph_contracted, pos=positions, with_labels=False, node_size=50, node_color=colors)
        plt.xlim((0, background.shape[0]))
        plt.ylim((0, background.shape[1]))
        plt.colorbar()

    def draw_schema(self, zmin=-3000, zmax=3000, title='', fname=None):
        """
        Рисуем граф Риба по трём уровням (минимумы, сёдла и максимумы).
        :param zmin:
        :param zmax:
        :param title:
        :param fname:
        :return:
        """
        fig = plt.figure(figsize=(6, 12))
        # ax = fig.add_subplot(111)
        ax = plt.gca()
        colors = list(nx.get_node_attributes(self.reeb_graph_contracted, 'color').values())
        positions = dict([(v, (self.critical_index[v], self.data[self.index[v][0], self.index[v][1]]))
                          for v in self.reeb_graph_contracted.nodes()])
        nx.draw_networkx_nodes(self.reeb_graph_contracted, pos=positions, with_labels=False, node_size=80,
                               node_color=colors, edgecolors='k', ax=ax)
        nx.draw_networkx_edges(self.reeb_graph_contracted, pos=positions, with_labels=False, node_size=80, ax=ax)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(zmin, zmax)
        plt.grid('on')

        # TODO: show ticks!

        if fname is not None:
            fig.savefig(fname)
            plt.close()

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
            ax.scatter(positions_x, positions_y, positions_z, c=colors, edgecolors='k', s=100)
        else:
            colors = list(nx.get_node_attributes(self.reeb_graph_contracted, 'color').values())
            positions_x = [self.index[idx][0] for idx in self.reeb_graph_contracted.nodes()]
            positions_y = [self.index[idx][1] for idx in self.reeb_graph_contracted.nodes()]
            positions_z = [self.data[self.index[idx][0], self.index[idx][1]] for idx in self.reeb_graph_contracted.nodes()]

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
        r = ReebGraph(data)
        r.make_index()
        r.cmp_merge_and_split_trees()
        r.convert_to_nx_graphs()
        r.cmp_augmented_reeb_graph()
        r.set_node_values_and_positions()
        r.check_graph_consistency()
        r.set_critical_index_and_colors()
        r.reeb_graph_contracted = r.reeb_graph_augmented.copy(as_view=False)
        r.contract_edges()
        r.set_persistence()
        return r


def test():
    import morse.field_generator
    data = np.array([
        [1.0, 2.0, 1.5],
        [2.5, 5.0, 1.6],
        [1.2, 2.2, 1.7],
        [2.5, 5.0, 1.6],
        [1.0, 2.0, 1.5]
    ])
    data = morse.field_generator.\
        gen_gaussian_sum_rectangle(50, 50, [(10, 10), (15, 15), (10, 15), (20, 5)], 3)
    r = ReebGraph.build_all(data)
    print(nx.get_node_attributes(r.reeb_graph_contracted, 'value'))
    print(nx.get_edge_attributes(r.reeb_graph_contracted, 'persistence'))

    # data = morse.field_generator.gen_sincos_field(200, 200, 0.3, 0.2)
    # data = morse.field_generator.gen_field_from_file("C:/data/test.bmp", conditions='plain')
    # data = morse.field_generator.gen_field_from_file(
    #     "C:/data/hmi/processed/AR12673/hmi_m_45s_2017_09_06_07_24_45_tai_magnetogram.fits",
    #     filetype='fits',
    #     conditions='plain')
    #

# test()