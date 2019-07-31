import importlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from morse.unionfind import UnionFind as UF

# Check that Pygraphviz is installed.
# If it is not, you cannot use dot program to draw Reeb graph.
pygraphviz_spec = importlib.util.find_spec("pygraphviz")
found_pygraphviz = pygraphviz_spec is not None


class ReebGraph:
    """
    Reeb Graph computation for data on quad mesh on the plane.
    We will compute Reeb Graph on ordered extended data,
    but simplify on initial data.
    """

    def __init__(self, initial_data):
        self.initial_order = ReebGraph._define_order(initial_data)

        w, h = initial_data.shape
        eps = 1.0 / (initial_data.size * 4)
        eps2 = eps ** 2

        self.data = np.zeros((w * 2 - 1, h * 2 - 1))
        self.data[0::2, 0::2] = initial_data

        ext_order = np.zeros((w * 2 - 1, h * 2 - 1))
        # "+1" because order contains zeros.
        ext_order[0::2, 0::2] = self.initial_order + 1

        # Рёбра одной ориентации
        for i in range(w - 1):
            for j in range(h):
                left_n, right_n = ext_order[2 * i, 2 * j], ext_order[2 * i + 2, 2 * j]
                left_v, right_v = self.data[2 * i, 2 * j], self.data[2 * i + 2, 2 * j]
                min_n, max_n = min(left_n, right_n), max(left_n, right_n)
                ext_order[2 * i + 1, 2 * j] = max_n + min_n * eps
                self.data[2 * i + 1, 2 * j] = max(left_v, right_v)

        # Рёбра другой ориентации
        for i in range(w):
            for j in range(h - 1):
                bot_n, top_n = ext_order[2 * i, 2 * j], ext_order[2 * i, 2 * j + 2]
                bot_v, top_v = self.data[2 * i, 2 * j], self.data[2 * i, 2 * j + 2]
                min_n, max_n = min(bot_n, top_n), max(bot_n, top_n)

                # "+1" because order contains zeros.
                ext_order[2 * i, 2 * j + 1] = max_n + min_n * eps
                self.data[2 * i, 2 * j + 1] = max(bot_v, top_v)

        # Грани
        for i in range(w - 1):
            for j in range(h - 1):
                verts = [ext_order[2 * i, 2 * j], ext_order[2 * i, 2 * j + 2],
                         ext_order[2 * i + 2, 2 * j + 2], ext_order[2 * i + 2, 2 * j]]
                verts_v = [self.data[2 * i, 2 * j], self.data[2 * i, 2 * j + 2],
                           self.data[2 * i + 2, 2 * j + 2], self.data[2 * i + 2, 2 * j]]
                verts.sort(reverse=True)

                # "+1" because order contains zeros.
                ext_order[2 * i + 1, 2 * j + 1] = verts[0] + verts[1] * eps + verts[2] * eps2
                self.data[2 * i + 1, 2 * j + 1] = max(verts_v)

        self.order = ReebGraph._define_order(ext_order)

        self.shape = self.data.shape
        self.index = None  # Ячейки в порядке возрастания значений
        self.cell_num = self.data.size

        self.merge_tree = []
        self.split_tree = []

        self.split_graph = nx.DiGraph()
        self.merge_graph = nx.DiGraph()
        self.reeb_graph_augmented = nx.DiGraph()
        self.reeb_graph_contracted = nx.DiGraph()

    @staticmethod
    def _define_order(data):
        order = np.zeros(data.size, dtype=np.int)
        order[data.flatten().argsort()] = np.arange(0, data.size, dtype=np.int)
        order = order.reshape(data.shape)
        return order

    @staticmethod
    def _reduce_node(graph, node, set_persistence=False):
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

    def _reduce_edge(self, e, verbose=False):
        """
        Contract edge in graph.
        We allow only saddle-extrema edge contraction.
        We check that after removing of extremum there are one downward and one upward (at least) direction.
        """
        graph = self.reeb_graph_contracted
        if graph.node[e[0]]['morse_index'] == 0:
            if verbose:
                print('Minimum ', e[0])
                print('Saddle ', e[1])

            if graph.out_degree(e[0]) == 1 and graph.in_degree(e[1]) >= 2:
                graph.remove_node(e[0])
                if verbose:
                    print('Remove', e[0])
                return True
            else:
                return False
        elif graph.node[e[1]]['morse_index'] == 2:
            print('Maximum ', e[1])
            print('Saddle ', e[0])

            if graph.in_degree(e[1]) == 1 and graph.out_degree(e[0]) >= 2:
                graph.remove_node(e[1])
                print('Remove', e[1])
                return True
            else:
                return False
        else:
            return False

    def _neighbours(self, i, j):
        """
        Get list of neighbors.
        We use topology of plane (R^2) without boundary conditions.
        """
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
                
    def _is_edge_contractible(self, v1, v2):
        g = self.reeb_graph_contracted
        idx1, idx2 = g.nodes[v1]['morse_index'], g.nodes[v2]['morse_index']
        
        # We cannot remove saddle-saddle pair;
        # also we cannot remove lonely minimum or lonely maximum.
        return not((idx1 == 1 and idx2 == 1) or
                   (idx1 == 0 and g.in_degree(v2) == 1) or
                   (idx2 == 2 and g.out_degree(v1) == 1))

    def make_index(self):
        """
        Sort cells by its values.
        :return: List of pairs [x, y] of array cells sorted by values.
        """
        y, x = np.meshgrid(range(self.shape[1]), range(self.shape[0]))
        flatten_data = self.order.flatten()
        self.index = np.dstack((x.flatten()[flatten_data.argsort()],
                                y.flatten()[flatten_data.argsort()]))[0]

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

    def set_morse_index(self):
        morse_index = dict([(node, -1) for node in self.reeb_graph_augmented.nodes])
        
        in_degree = self.reeb_graph_augmented.in_degree
        out_degree = self.reeb_graph_augmented.out_degree

        for i in self.reeb_graph_augmented.nodes:
            if in_degree[i] == 0:
                morse_index[i] = 0
            elif out_degree[i] == 0:
                morse_index[i] = 2
            else: 
                morse_index[i] = 1
        nx.set_node_attributes(self.reeb_graph_augmented, morse_index, 'morse_index')

    def cmp_augmented_reeb_graph(self):
        """
        See Carr, H., Snoeyink, J., & Axen, U. (2003).
        Computing contour trees in all dimensions.
        Computational Geometry, 24(2), 75–94.
        """
        mg = self.merge_graph.copy()
        sg = self.split_graph.copy()
        
        # 1. For each vertex xi, if up-degree in join tree + down-degree in split tree is 1, enqueue x_i.
        leaf_queue = deque(reversed([x for x in range(self.cell_num) if
                                     mg.in_degree(x) + sg.out_degree(x) == 1]))

        # 2. Initialize contour tree to an empty graph on ||join tree|| vertices.
        self.reeb_graph_augmented = nx.DiGraph()
        self.reeb_graph_augmented.add_nodes_from(range(self.cell_num))

        # 3. While leaf queue size > 1
        while leaf_queue:
            # Dequeue the first vertex, xi, on the leaf queue.
            x = leaf_queue.pop()

            # If xi is an upper leaf
            if mg.in_degree[x] == 0:
                if sg.out_degree[x] == 0:
                    break  # Вершина без рёбер

                # find incident arc y_i y_j in join tree.
                neighbour = list(mg.successors(x))[0]
                self.reeb_graph_augmented.add_edge(x, neighbour)
            else:
                # else find incident arc z_iz_j in split tree.
                neighbour = list(sg.predecessors(x))[0]
                self.reeb_graph_augmented.add_edge(neighbour, x)

            # Remove node from split and merge graphs
            # See Definition 4.5 from H. Carr (2003)
            self._reduce_node(mg, x)
            self._reduce_node(sg, x)

            # If x_j is now a leaf, enqueue x_j
            if mg.in_degree(neighbour) + sg.out_degree(neighbour) == 1:
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
        
    def draw_reebgraph(self, annotate=True):
        """
        Draw contracted Reeb Graph.
        This method uses dot layout, so GraphViz dot and pygraphviz are required.
        """        
        if not found_pygraphviz:
            print("You cannot draw Reeb graph without Graphviz and Pygraphviz libraries!")
            return
            
        g = self.reeb_graph_contracted
        positions = nx.nx_agraph.pygraphviz_layout(g, prog='dot')
        morse_index = list(nx.get_node_attributes(g, 'morse_index').values())
        color_index = [(0, 0, 1), (0, 0.5, 0), (1, 0, 0)]
        colors = [color_index[i] for i in morse_index]
        nx.draw(g, positions, node_size=50, node_color=colors, with_labels=False)
        if annotate:
            nx.draw_networkx_labels(g, positions, labels=dict(list(zip(list(g.nodes), list(map(str, list(g.nodes)))))))

    def draw2d(self, background=None, vmin=-2000, vmax=2000):
        """
        :param background: background image (By default, self.data)
        """
        if background is None:
            background = self.data
        plt.matshow(background, cmap='gray', vmin=vmin, vmax=vmax)
        morse_index = list(nx.get_node_attributes(self.reeb_graph_contracted, 'morse_index').values())
        color_index = [(0, 0, 1), (0, 0.5, 0), (1, 0, 0)]
        colors = [color_index[i] for i in morse_index]
        positions = nx.get_node_attributes(self.reeb_graph_contracted, 'position')
        nx.draw_networkx(self.reeb_graph_contracted, pos=positions, with_labels=False, node_size=50, node_color=colors)
        plt.xlim((0, background.shape[1]))
        plt.ylim((0, background.shape[0]))
        plt.colorbar()

    def draw_schema(self, zmin=-3000, zmax=3000, title=''):
        """
        Draw 3-level Reeb Graph.
        """
        fig = plt.figure(figsize=(6, 12))
        # ax = fig.add_subplot(111) 
        ax = plt.gca()
        morse_index = list(nx.get_node_attributes(self.reeb_graph_contracted, 'morse_index').values())
        color_index = [(0, 0, 1), (0, 0.5, 0), (1, 0, 0)]
        colors = [color_index[i] for i in morse_index]
        positions = dict([(v, (self.critical_index[v], self.data[self.index[v][0], self.index[v][1]]))
                          for v in self.reeb_graph_contracted.nodes()])
        nx.draw_networkx_nodes(self.reeb_graph_contracted, pos=positions, with_labels=False, node_size=80,
                               node_color=colors, edgecolors='k', ax=ax)
        nx.draw_networkx_edges(self.reeb_graph_contracted, pos=positions, with_labels=False, node_size=80, ax=ax)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(zmin, zmax)
        plt.grid('on')  # TODO: show ticks!

    def draw3d(self, draw_edges=False, use_threshold=True, threshold=20, xmin=0, ymin=0, xmax=None, ymax=None,
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

    def persistence_simplification(self, level):
        """
        Persistence simplification of Reeb graph.
        :param level: level of simplification.
        """
        import heapq
        
        g = self.reeb_graph_contracted
        edge_persistence = nx.get_edge_attributes(g, 'persistence')
        
        # Create Priority Queue of contractable edges.
        # Main principle: If edge was not contractible it will not be contractible.
        heap = [(edge_persistence[e], e) for e in g.edges if self._is_edge_contractible(e[0], e[1])]

        if not heap:  # If heap is empty, then there's nothing to simplify.
            return

        heapq.heapify(heap)
        
        curr_persistence, curr_edge = heapq.heappop(heap)

        while curr_persistence < level:
            # Some edges are already removed.
            # But we cannot remove edge DIRECTLY from heap (it's inefficient)
            if g.has_edge(*curr_edge):
                if self._is_edge_contractible(*curr_edge):
                    # plt.figure()
                    # self.draw_reebgraph()
                    # plt.show()
                    if g.nodes[curr_edge[1]]['morse_index'] == 1:
                        # Remove maximum
                        g.remove_node(curr_edge[0])
                        saddle = curr_edge[1]
                    else:
                        # Remove minimum
                        g.remove_node(curr_edge[1])
                        saddle = curr_edge[0]
                        
                    # Check whether we can remove saddle.
                    # In general - we can. But there exist degenerations.
                    if g.in_degree(saddle) == 1 and g.out_degree(saddle) == 1:
                        # Если у седла входящая и выходящая степени равны 1, то
                        # удаляем седло и стягиваем его соседей в ребро.
                        prev_neighbour = list(g.predecessors(saddle))[0]
                        next_neighbour = list(g.successors(saddle))[0]
                        g.remove_node(saddle)
                        
                        new_edge_persistence = g.nodes[next_neighbour]['value'] - g.nodes[prev_neighbour]['value']
                        g.add_edge(prev_neighbour, next_neighbour, persistence=new_edge_persistence)
                        
                        if self._is_edge_contractible(prev_neighbour, next_neighbour):
                            heapq.heappush(heap, (new_edge_persistence, (prev_neighbour, next_neighbour)))
            if not heap:
                return
            curr_persistence, curr_edge = heapq.heappop(heap)
            
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
        r.set_morse_index()
        r.reeb_graph_contracted = r.reeb_graph_augmented.copy(as_view=False)
        r.contract_edges()
        r.set_persistence()
        return r


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
    # data = morse.field_generator.perturb(data)
    im = morse.field_generator.gen_field_from_file('C:/data/test.fits', filetype='fits', conditions='plain')
    r = ReebGraph.build_all(im)
    # print(nx.get_node_attributes(r.reeb_graph_contracted, 'value'))
    # print(nx.get_edge_attributes(r.reeb_graph_contracted, 'persistence'))

    r.persistence_simplification(1000)

    print(len(r.reeb_graph_contracted.nodes))
    plt.figure()
    r.draw_reebgraph()
    plt.show()

    # data = morse.field_generator.gen_sincos_field(200, 200, 0.3, 0.2)
    # data = morse.field_generator.gen_field_from_file("C:/data/test.bmp", conditions='plain')
    # data = morse.field_generator.gen_field_from_file(
    #     "C:/data/hmi/processed/AR12673/hmi_m_45s_2017_09_06_07_24_45_tai_magnetogram.fits",
    #     filetype='fits',
    #     conditions='plain')
    #

# test()