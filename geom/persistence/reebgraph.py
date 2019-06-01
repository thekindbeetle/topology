import numpy as np
import networkx as nx
from collections import deque


class ReebGraph:
    data = None  # Исходные данные
    index = None  # Ячейки в порядке возрастания значений
    cell_num = 0
    shape = []

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

    def _neighbors(self, i, j):
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
        x, y = np.meshgrid(range(self.shape[0]), range(self.shape[1]))
        flatten_data = self.data.flatten()
        self.index = np.dstack((x.flatten()[np.argsort(flatten_data)],
                                y.flatten()[np.argsort(flatten_data)]))[0]

    def cmp_merge_and_split_graphs(self):
        # Compute merge tree
        label = -np.ones(self.shape, dtype=np.int)
        for idx in range(self.cell_num):
            neighbour_labels = [label[i[0], i[1]] for i in self._neighbors(*self.index[idx])]
            neighbour_labels = np.unique(neighbour_labels)
            neighbour_labels = neighbour_labels[neighbour_labels >= 0]
            if len(neighbour_labels) == 0:
                # If cell has no labelled neighbors, component created
                self.critical_points.append(idx)
                self.critical_points_idx.append(0)  # Помечаем минимумом
                label[self.index[idx][0], self.index[idx][1]] = idx
            elif len(neighbour_labels) == 1:
                label[self.index[idx][0], self.index[idx][1]] = neighbour_labels[0]
            elif len(neighbour_labels) == 2:
                self.critical_points.append(idx)
                self.critical_points_idx.append(1)  # Помечаем седлом
                self.merge_tree.append((neighbour_labels[0], idx))
                self.merge_tree.append((neighbour_labels[1], idx))
                label[label == neighbour_labels[0]] = idx
                label[label == neighbour_labels[1]] = idx
                label[self.index[idx][0], self.index[idx][1]] = idx
            else:
                self.critical_points.append(idx)
                self.critical_points_idx.append(1)  # Помечаем седлом
                for neighbour_label in neighbour_labels:
                    self.merge_tree.append((neighbour_label, idx))
                    label[label == neighbour_label] = idx
                label[self.index[idx][0], self.index[idx][1]] = idx

        # Compute split tree
        label = -np.ones(self.shape, dtype=np.int)
        for idx in reversed(range(self.cell_num)):
            neighbour_labels = [label[i[0], i[1]] for i in self._neighbors(*self.index[idx])]
            neighbour_labels = np.unique(neighbour_labels)
            neighbour_labels = neighbour_labels[neighbour_labels >= 0]
            if len(neighbour_labels) == 0:
                # If cell has no labelled neighbors, component created
                self.critical_points.append(idx)
                self.critical_points_idx.append(2)  # Помечаем максимумом
                label[self.index[idx][0], self.index[idx][1]] = idx
            elif len(neighbour_labels) == 1:
                label[self.index[idx][0], self.index[idx][1]] = neighbour_labels[0]
            elif len(neighbour_labels) == 2:
                self.critical_points.append(idx)
                self.critical_points_idx.append(1)  # Помечаем седлом
                self.split_tree.append((idx, neighbour_labels[0]))
                self.split_tree.append((idx, neighbour_labels[1]))
                label[label == neighbour_labels[0]] = idx
                label[label == neighbour_labels[1]] = idx
                label[self.index[idx][0], self.index[idx][1]] = idx
            else:
                self.critical_points.append(idx)
                self.critical_points_idx.append(1)  # Помечаем седлом
                for neighbour_label in neighbour_labels:
                    self.merge_tree.append((idx, neighbour_label))
                    label[label == neighbour_label] = idx
                label[self.index[idx][0], self.index[idx][1]] = idx
                print("Дохрена меток!")
        # TODO: Если только один максимум или один минимум, то теряем рёбра.

    def convert_to_nx_graphs(self):
        self.split_graph = nx.DiGraph()
        self.merge_graph = nx.DiGraph()
        self.split_graph.add_nodes_from(self.critical_points)
        self.merge_graph.add_nodes_from(self.critical_points)
        self.split_graph.add_edges_from(self.split_tree)
        self.merge_graph.add_edges_from(self.merge_tree)

    def cmp_reeb_graph(self):
        """
        See Carr, H., Snoeyink, J., & Axen, U. (2003).
        Computing contour trees in all dimensions.
        Computational Geometry, 24(2), 75–94.

        Здесь ситуация проще, циклов нет.
        Просто склеиваем два графа (split and merge).
        """
        self.reeb_graph = nx.DiGraph()
        self.reeb_graph.add_nodes_from(self.critical_points)
        self.reeb_graph.add_edges_from(self.split_tree)
        self.reeb_graph.add_edges_from(self.merge_tree)

        # Запишем индексы критических точек в вершины графа
        nx.set_node_attributes(self.reeb_graph,
                               dict(zip(self.critical_points, self.critical_points_idx)),
                               "morse_index")

        # Запишем значения персистентности в рёбра графа
        persistence = dict([(edge, self.data[self.index[edge[1]]] - self.data[self.index[edge[0]]])
                            for edge in self.reeb_graph.edges])
        print([edge for edge in self.reeb_graph.edges])
        nx.set_edge_attributes(self.reeb_graph, persistence, "persistence")


def test():
    import morse.field_generator
    import matplotlib.pyplot as plt
    data = np.array([
        [1.0, 2.0, 1.5],
        [2.5, 5.0, 1.6],
        [1.2, 2.2, 1.7],
        [2.5, 5.0, 1.6],
        [1.0, 2.0, 1.5]
    ])
    # data = morse.field_generator.\
    #     gen_gaussian_sum_rectangle(50, 50, [(10, 10), (15, 15), (10, 15), (20, 5)], 3)

    # data = morse.field_generator.gen_sincos_field(200, 200, 0.3, 0.2)
    # data = morse.field_generator.gen_field_from_file("C:/data/1.bmp", conditions='plain')
    # data = morse.field_generator.perturb(data)
    #
    reeb = ReebGraph(data)
    reeb.make_index()
    reeb.cmp_merge_and_split_graphs()
    reeb.convert_to_nx_graphs()
    reeb.cmp_reeb_graph()

    labels_dict = dict([(idx, idx) for idx in reeb.critical_points])
    positions = dict([(reeb.critical_points[i], reeb.index[reeb.critical_points[i]])
                      for i in range(len(reeb.critical_points))])

    plt.matshow(np.transpose(data))
    # nx.draw(reeb.reeb_graph, pos=positions, labels=labels_dict)
    nx.draw(reeb.merge_graph, pos=positions, labels=labels_dict)
    nx.draw_networkx_edge_labels(reeb.reeb_graph,
                                 pos=positions,
                                 edge_labels=nx.get_edge_attributes(reeb.reeb_graph, 'persistence'))

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
