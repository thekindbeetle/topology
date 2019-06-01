import numpy as np
import networkx as nx
from collections import deque
from morse.unionfind import UnionFind as UF


class ReebGraph:
    data = None  # Исходные данные
    index = None  # Ячейки в порядке возрастания значений
    order = None  # Номера ячеек на сетке
    cell_num = 0
    shape = []
    colors = []

    critical_points = []
    critical_points_idx = []

    merge_tree = []
    split_tree = []

    split_graph = nx.DiGraph()
    merge_graph = nx.DiGraph()
    reeb_graph = nx.DiGraph()

    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.cell_num = data.size
        self.order = np.zeros(data.shape)

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

    def set_colors(self):
        self.colors = [(0, 0, 0) for i in range(self.cell_num)]
        for i in range(self.cell_num):
            if (self.reeb_graph.in_degree(i) == 0) and (self.reeb_graph.out_degree(i) != 0):
                self.colors[i] = (0, 0, 1)
            if (self.reeb_graph.in_degree(i) != 0) and (self.reeb_graph.out_degree(i) == 0):
                self.colors[i] = (1, 0, 0)

    @staticmethod
    def _reduce_node(graph, node):
        """
        Remove node from graph.
        Connect neighbour vertices.
        """
        if graph.in_degree(node) == 1 and graph.out_degree(node) == 1:
            prev_neighbour = list(graph.predecessors(node))[0]
            next_neighbour = list(graph.successors(node))[0]
            graph.add_edge(prev_neighbour, next_neighbour)
        graph.remove_node(node)

    def cmp_reeb_graph(self):
        """
        See Carr, H., Snoeyink, J., & Axen, U. (2003).
        Computing contour trees in all dimensions.
        Computational Geometry, 24(2), 75–94.
        """
        # 1. For each vertex xi, if up-degree in join tree + down-degree in split tree is 1, enqueue x_i.
        leaf_queue = deque(reversed([x for x in range(self.cell_num) if
                                     self.merge_graph.in_degree(x) + self.split_graph.out_degree(x) == 1]))

        # 2. Initialize contour tree to an empty graph on ||join tree|| vertices.
        self.reeb_graph = nx.DiGraph()
        self.reeb_graph.add_nodes_from(range(self.cell_num))

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
                self.reeb_graph.add_edge(x, neighbour)
            else:
                # else find incident arc z_iz_j in split tree.
                neighbour = list(self.split_graph.predecessors(x))[0]
                self.reeb_graph.add_edge(neighbour, x)

            # Remove node from split and merge graphs
            # See Definition 4.5 from H. Carr (2003)
            self._reduce_node(self.merge_graph, x)
            self._reduce_node(self.split_graph, x)

            # If x_j is now a leaf, enqueue x_j
            if self.merge_graph.in_degree(neighbour) + self.split_graph.out_degree(neighbour) == 1:
                leaf_queue.append(neighbour)


def test():
    import morse.field_generator
    import matplotlib.pyplot as plt
    # data = np.array([
    #     [1.0, 2.0, 1.5],
    #     [2.5, 5.0, 1.6],
    #     [1.2, 2.2, 1.7],
    #     [2.5, 5.0, 1.6],
    #     [1.0, 2.0, 1.5]
    # ])
    data = morse.field_generator.\
        gen_gaussian_sum_rectangle(50, 50, [(10, 10), (15, 15), (10, 15), (20, 5)], 3)

    # data = morse.field_generator.gen_sincos_field(200, 200, 0.3, 0.2)
    # data = morse.field_generator.gen_field_from_file("C:/data/1.bmp", conditions='plain')
    data = morse.field_generator.perturb(data)

    reeb = ReebGraph(data)
    reeb.make_index()
    reeb.cmp_merge_and_split_trees()
    reeb.convert_to_nx_graphs()
    reeb.cmp_reeb_graph()

    # labels_dict = dict([(idx, idx) for idx in reeb.critical_points])
    # positions = dict([(reeb.critical_points[i], reeb.index[reeb.critical_points[i]])
    #                   for i in range(len(reeb.critical_points))])

    # plt.matshow(np.transpose(data))
    # # nx.draw(reeb.reeb_graph, pos=positions, labels=labels_dict)
    # nx.draw(reeb.merge_graph, pos=positions, labels=labels_dict)
    # nx.draw_networkx_edge_labels(reeb.reeb_graph,
    #                              pos=positions,
    #                              edge_labels=nx.get_edge_attributes(reeb.reeb_graph, 'persistence'))

    # x = [reeb.index[reeb.cpoints[i]][0] for i in range(len(reeb.cpoints)) if reeb.cpoints_idx[i] == 0]
    # y = [reeb.index[reeb.cpoints[i]][1] for i in range(len(reeb.cpoints)) if reeb.cpoints_idx[i] == 0]
    # plt.scatter(x, y, c='b')
    # x = [reeb.index[reeb.cpoints[i]][0] for i in range(len(reeb.cpoints)) if reeb.cpoints_idx[i] == 1]
    # y = [reeb.index[reeb.cpoints[i]][1] for i in range(len(reeb.cpoints)) if reeb.cpoints_idx[i] == 1]
    # plt.scatter(x, y, c='g')
    # x = [reeb.index[reeb.cpoints[i]][0] for i in range(len(reeb.cpoints)) if reeb.cpoints_idx[i] == 2]
    # y = [reeb.index[reeb.cpoints[i]][1] for i in range(len(reeb.cpoints)) if reeb.cpoints_idx[i] == 2]
    # plt.scatter(x, y, c='r')

    # for edge in reeb.merge_tree:
    #     plt.arrow(reeb.index[edge[0]][0], reeb.index[edge[0]][1],
    #               reeb.index[edge[1]][0] - reeb.index[edge[0]][0],
    #               reeb.index[edge[1]][1] - reeb.index[edge[0]][1], fc="k", ec="k",
    #               head_width=1, head_length=1.5, capstyle='round')
    #
    # for edge in reeb.split_tree:
    #     plt.arrow(reeb.index[edge[0]][0], reeb.index[edge[0]][1],
    #               reeb.index[edge[1]][0] - reeb.index[edge[0]][0],
    #               reeb.index[edge[1]][1] - reeb.index[edge[0]][1], fc="r", ec="r",
    #               head_width=1, head_length=1.5, capstyle='round')



    # print(reeb.merge_tree)
    # print(reeb.split_tree)
    # print(reeb.cpoints)

    # plt.matshow(data)
    #
    # x = [reeb.index[reeb.cpoints[i]][0] for i in range(len(reeb.cpoints)) if reeb.cpoints_idx[i] == 0]
    # y = [reeb.index[reeb.cpoints[i]][1] for i in range(len(reeb.cpoints)) if reeb.cpoints_idx[i] == 0]
    # plt.scatter(x, y, c='b')
    # x = [reeb.index[reeb.cpoints[i]][0] for i in range(len(reeb.cpoints)) if reeb.cpoints_idx[i] == 1]
    # y = [reeb.index[reeb.cpoints[i]][1] for i in range(len(reeb.cpoints)) if reeb.cpoints_idx[i] == 1]
    # plt.scatter(x, y, c='g')
    # x = [reeb.index[reeb.cpoints[i]][0] for i in range(len(reeb.cpoints)) if reeb.cpoints_idx[i] == 2]
    # y = [reeb.index[reeb.cpoints[i]][1] for i in range(len(reeb.cpoints)) if reeb.cpoints_idx[i] == 2]
    # plt.scatter(x, y, c='r')

test()
