import pygame
import sys
import pickle
# from FactoryObjects.Factory import Factory
from FactoryObjects.Path import Path
# from FactoryObjects.AGV import AGV
# from FactoryObjects.LoadingStation import LoadingStation
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import networkx as nx

class FabricPathGraph:
    def __init__(self, factory):
        self.factory = factory
        self.grid = None
        self.nodes = []
        self.object_nodes = []
        self.crossing_nodes = []
        self.agv_nodes = dict()
        self.horizontal_paths = []
        self.vertical_paths = []
        self.object_paths = []
        self.agv_paths = []
        self.layout_paths = []
        self.only_free_agv = False

        self.adjacency_matrix = None
        self.dijkstra_matrix = None
        self.predecessors = None

        self.node_index_mapping = None
        self.index_node_mapping = None
        self.object_index_mapping = None

        self.graph_screen = None
        self.pixel_size = 25
        self.color_grid = factory.get_color_grid()

        self.write_files = False

    def construct_graph(self):
        self.grid = self.factory.factory_grid_layout
        self.agv_nodes = dict()
        self.object_paths = []
        self.vertical_paths = []
        self.horizontal_paths = []
        self.agv_paths = []
        self.create_object_nodes()
        self.nodes = self.create_paths()
        self.add_agvs()
        # layout paths without agv path since these are handled separately
        self.layout_paths = self.vertical_paths + self.object_paths + self.horizontal_paths

    def view_graph(self):
        pygame.init()
        self.graph_screen = pygame.display.set_mode((self.factory.length * self.pixel_size, self.factory.width * self.pixel_size))
        pygame.display.set_caption('Path-Graph')
        self.display_colors()

        for node in self.nodes:
            pygame.draw.circle(self.graph_screen, color=(63, 37, 229), center=(node[0] * self.pixel_size + self.pixel_size / 2, node[1] * self.pixel_size + self.pixel_size / 2), radius=8,
                               width=25)
            pygame.display.flip()
            time.sleep(0.01)

        for node in self.agv_nodes.values():
            pygame.draw.circle(self.graph_screen, color=(255, 128, 0), center=(node[0] * self.pixel_size + self.pixel_size / 7, node[1] * self.pixel_size + self.pixel_size / 7),
                               radius=8,
                               width=25)
            pygame.display.flip()

        for path in self.layout_paths:
            pygame.draw.line(self.graph_screen, color=(63, 37, 229),
                             start_pos=[path[0][0] * self.pixel_size + self.pixel_size / 2, path[0][1] * self.pixel_size + self.pixel_size / 2],
                             end_pos=[path[1][0] * self.pixel_size + self.pixel_size / 2, path[1][1] * self.pixel_size + self.pixel_size / 2], width=4)
            pygame.display.flip()
            time.sleep(0.01)

        for path in self.agv_paths:
            pygame.draw.line(self.graph_screen, color=(255, 128, 0),
                             start_pos=[path[0][0] * self.pixel_size + self.pixel_size / 8, path[0][1] * self.pixel_size + self.pixel_size / 8],
                             end_pos=[path[1][0] * self.pixel_size + self.pixel_size / 5, path[1][1] * self.pixel_size + self.pixel_size / 5], width=4)
            pygame.display.flip()
            time.sleep(0.01)

        opened = True
        while opened:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    opened = False
                    pygame.quit()
                    break
                    # sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        opened = False
                        break
                        # sys.exit()
    #
    # 50 = 50
    # self.height = self.factory.width
    # self.width = self.factory.length
    # self.reference_size = self.factory.cell_size * 1000
    # self.screen = None

    def display_colors(self):
        reference_size = self.factory.cell_size * 1000
        # Draw the color data
        for y in range(self.factory.width):
            for x in range(self.factory.length):
                pygame.draw.rect(self.graph_screen, self.color_grid[x][y], (x * self.pixel_size, y * self.pixel_size, self.pixel_size, self.pixel_size))

        # Draw the AGVs
        for agv in self.factory.agvs:
            agv_pos = [agv.pos_x, agv.pos_y]
            pygame.draw.rect(self.graph_screen, [0, 0, 0],
                             (agv_pos[0] * self.pixel_size + (agv.width / reference_size * self.pixel_size) / 2,
                              agv_pos[1] * self.pixel_size + (agv.width / reference_size * self.pixel_size) / 2,
                              agv.width / reference_size * self.pixel_size,
                              agv.length / reference_size * self.pixel_size),
                             border_radius=3)

        pygame.display.flip()

    def create_object_nodes(self):
        self.object_nodes = []
        for warehouse in self.factory.warehouses:
            # self.object_nodes += [staple.get_position() for staple in warehouse.input_staples + warehouse.output_staples]
            self.object_nodes.append(tuple(warehouse.pos_input))
            self.object_nodes.append(tuple(warehouse.pos_output))
        for machine in self.factory.machines:
            # self.object_nodes += [staple.get_position() for staple in machine.input_staples + machine.output_staples]
            self.object_nodes.append(tuple(machine.pos_input))
            self.object_nodes.append(tuple(machine.pos_output))
        for loading_station in self.factory.loading_stations:
            self.object_nodes.append((loading_station.pos_x, loading_station.pos_y))

    def create_paths(self):
        self.crossing_nodes = []
        for c, col in enumerate(self.grid):
            for r, cell in enumerate(col):
                if type(cell) is Path:
                    is_crossing, from_left, from_top = self.cell_is_crossing(c, r)
                    if is_crossing:
                        self.crossing_nodes.append((c, r))
                    if from_left:
                        self.add_path((c, r), horizontal=True)
                    if from_top:
                        self.add_path((c, r), horizontal=False)
        return self.object_nodes + self.crossing_nodes

    def cell_is_crossing(self, column: int, row: int):

        max_col = len(self.grid) - 1
        max_row = len(self.grid[0]) - 1
        horizontal = False
        vertical = False
        from_left = False
        from_top = False
        from_bot = False
        from_right = False
        object_neighbor = False

        # look top
        if row > 0:
            if (column, row - 1) in self.object_nodes:
                object_neighbor = True
                self.object_paths.append(((column, row - 1), (column, row)))
            elif type(self.grid[column][row - 1]) is Path:
                from_top = True

        # look left
        if column > 0:
            if (column - 1, row) in self.object_nodes:
                object_neighbor = True
                self.object_paths.append(((column - 1, row), (column, row)))

            elif type(self.grid[column - 1][row]) is Path:
                from_left = True

        # look down
        if row < max_row:
            if (column, row + 1) in self.object_nodes:
                object_neighbor = True
                self.object_paths.append(((column, row), (column, row + 1)))

            elif type(self.grid[column][row + 1]) is Path:
                from_bot = True


        # look right
        if column < max_col:
            if (column + 1, row) in self.object_nodes:
                object_neighbor = True
                self.object_paths.append(((column, row), (column + 1, row)))

            elif type(self.grid[column + 1][row]) is Path:
                from_right = True

        if all([from_top, from_bot]) != all([from_left, from_right]) and not object_neighbor:
            if  sum([from_bot, from_top, from_right, from_left]) % 2 == 1:
                is_crossing = True
            else:
                is_crossing = False
        else:
            is_crossing = True

        # only return left and top for path building since we iterate from top-left to bottom_right
        return is_crossing, from_left, from_top

    def add_path(self, cell: (int, int), horizontal: bool):
        found = False

        if horizontal:
            for i, path in enumerate(self.horizontal_paths):
                if path[1] == (cell[0] - 1, cell[1]) and (cell[0] - 1, cell[1]) not in self.crossing_nodes:
                    self.horizontal_paths[i] = (path[0], cell)
                    found = True
        else:
            for i, path in enumerate(self.vertical_paths):
                if path[1] == (cell[0], cell[1] - 1) and (cell[0], cell[1] - 1) not in self.crossing_nodes:
                    self.vertical_paths[i] = (path[0], cell)
                    found = True

        if not found:
            if horizontal:
                self.horizontal_paths.append(((cell[0] - 1, cell[1]), (cell[0], cell[1])))
            else:
                self.vertical_paths.append(((cell[0], cell[1] - 1), (cell[0], cell[1])))

    def add_agvs(self):
        for agv in self.factory.agvs:
            if self.only_free_agv:
                if agv.is_free:
                    if agv.coupling_master is None:
                        if (type(self.grid[int(agv.pos_x)][int(agv.pos_y)]) is not Path and
                                (int(agv.pos_x), int(agv.pos_y)) not in self.object_nodes):
                            self.agv_nodes[agv] = self.find_adjacent_object_node((int(agv.pos_x), int(agv.pos_y)))
                        else:
                            self.agv_nodes[agv] = (int(agv.pos_x) + 0.1, int(agv.pos_y) + 0.1)

                    else:
                        self.agv_nodes[agv] = (int(agv.coupling_master.pos_x) + 0.1, int(agv.coupling_master.pos_y) + 0.1)
            else:
                if agv.coupling_master is None:
                    if (type(self.grid[int(agv.pos_x)][int(agv.pos_y)]) is not Path and
                            (int(agv.pos_x), int(agv.pos_y)) not in self.object_nodes):
                        self.agv_nodes[agv] = self.find_adjacent_object_node((int(agv.pos_x), int(agv.pos_y)))
                    else:
                        self.agv_nodes[agv] = (int(agv.pos_x) + 0.1, int(agv.pos_y) + 0.1)
                else:
                    self.agv_nodes[agv] = (int(agv.coupling_master.pos_x) + 0.1, int(agv.coupling_master.pos_y) + 0.1)

        for node in self.agv_nodes.values():
            done = False
            connection_node = (int(round(node[0] - 0.1)), int(round(node[1] - 0.1)))
            if connection_node in self.nodes:
                self.layout_paths.append((connection_node, node))
                self.agv_paths.append((connection_node, node))
            else:
                for i, path in enumerate(self.horizontal_paths):
                    if (path[0][1] == connection_node[1] and path[0][0]< connection_node[0] and
                            path[1][0] > connection_node[0]):
                        self.nodes.append(connection_node)
                        self.horizontal_paths[i] = (path[0], connection_node)
                        self.horizontal_paths.append((connection_node, path[1]))
                        self.agv_paths.append((connection_node, node))
                        done = True
                        break
                if done:
                    continue
                for i, path in enumerate(self.vertical_paths):
                    if (path[0][0] == connection_node[0] and path[0][1] < connection_node[1] and
                            path[1][1] > connection_node[1]):
                        self.nodes.append(connection_node)
                        self.vertical_paths[i] = (path[0], connection_node)
                        self.vertical_paths.append((connection_node, path[1]))
                        self.agv_paths.append((connection_node, node))
                        break

    def find_adjacent_object_node(self, node):
        if (node[0] - 1, node[1]) in self.object_nodes:  # left
            return node[0] - 1, node[1]

        if (node[0], node[1] - 1) in self.object_nodes:  # top
            return node[0], node[1] - 1

        if (node[0] - 1, node[1] - 1 ) in self.object_nodes:  # top-left
            return node[0] - 1, node[1] - 1

        if (node[0], node[1] + 1) in self.object_nodes:  # bottom
            return node[0], node[1] + 1

        if (node[0] + 1, node[1]) in self.object_nodes:  # right
            return node[0] + 1, node[1]

        if (node[0] - 1, node[1] + 1) in self.object_nodes:  # bottom-left
            return node[0] - 1, node[1] + 1

        if (node[0] + 1, node[1] - 1) in self.object_nodes:  # top-right
            return node[0] + 1, node[1] - 1

        if (node[0] + 1, node[1] + 1) in self.object_nodes:  # bottom-right
            return node[0] + 1, node[1] + 1


        for agv in self.factory.agvs:
            print(agv.get_info())
        raise ValueError("Ein AGV hat ist außerhalb der Pfade und nicht um Umkreis von < 2 zu einem anderen objekt ")


    def create_distance_matrix(self):

        def distance(n1, n2):
            return sum(abs(val1 - val2) for val1, val2 in zip(n1, n2))

        dimension = len(self.nodes) + len(self.agv_nodes)

        self.node_index_mapping = dict()
        self.index_node_mapping = dict()
        self.object_index_mapping = dict()

        for i, tup in enumerate(self.agv_nodes.items()):
            self.object_index_mapping[tup[0]] = i
            self.index_node_mapping[i] = tup[1]

        for i, node in enumerate(self.nodes):
            self.node_index_mapping[node] = i + len(self.agv_nodes)
            self.index_node_mapping[i + len(self.agv_nodes)] = node
            self.object_index_mapping[self.grid[node[0]][node[1]]] = i + len(self.agv_nodes)

        self.adjacency_matrix = np.zeros(shape=(dimension, dimension))

        for path in self.layout_paths:
            if path[0] in self.object_nodes or path[1] in self.object_nodes:
                self.adjacency_matrix[self.node_index_mapping[path[0]]][self.node_index_mapping[path[1]]] = distance(path[0], path[1]) + 0.0001
            else:
                self.adjacency_matrix[self.node_index_mapping[path[0]]][self.node_index_mapping[path[1]]] = distance(path[0], path[1]) + 0.00001


        for agv_node, path in zip(self.agv_nodes.items(), self.agv_paths):
            self.adjacency_matrix[self.object_index_mapping[agv_node[0]]][self.node_index_mapping[path[0]]] = distance(agv_node[1], path[0]) + 0.00001

        csr_graph = csr_matrix(self.adjacency_matrix)

        self.dijkstra_matrix, self.predecessors = dijkstra(csgraph=csr_graph, directed=False, return_predecessors=True)

        for agv_node, path in zip(self.agv_nodes.items(), self.agv_paths):
            if path[0] in self.object_nodes:
                self.dijkstra_matrix[self.object_index_mapping[agv_node[0]]][self.node_index_mapping[path[0]]] += 0.00001

        if self.write_files:
            np.savetxt("graph.csv", self.adjacency_matrix,  delimiter="; ")
            np.savetxt("dijkstra_matrix.csv", self.dijkstra_matrix, delimiter="; ")
            np.savetxt("predecessors.csv", self.predecessors, delimiter="; ")

        relevant_indexes = []
        for agv in self.agv_nodes.keys():
            relevant_indexes.append(self.object_index_mapping[agv])
        #
        for wh in self.factory.warehouses:

            relevant_indexes.append(self.node_index_mapping[tuple(wh.pos_input)])
            relevant_indexes.append(self.node_index_mapping[tuple(wh.pos_output)])

        for machine in self.factory.machines:
            relevant_indexes.append(self.node_index_mapping[tuple(machine.pos_input)])
            relevant_indexes.append(self.node_index_mapping[tuple(machine.pos_output)])

        # for warehouse in self.factory.warehouses:
        #     relevant_indexes += [self.node_index_mapping[staple.get_position()] for staple in
        #                           warehouse.input_staples + warehouse.output_staples]
        #
        # for machine in self.factory.machines:
        #     relevant_indexes += [self.node_index_mapping[staple.get_position()] for staple in
        #                           machine.input_staples + machine.output_staples]


        distance_matrix = self.dijkstra_matrix[relevant_indexes, :]
        distance_matrix = distance_matrix[:, relevant_indexes]
        # print(f"Indexes: {relevant_indexes}")
        return distance_matrix

    def get_predecessor_queue(self, start, target):
        # self.construct_graph()
        _ = self.create_distance_matrix()
        if type(target) is not tuple:
            target = tuple(target)
        queue = []
        if start not in self.node_index_mapping:
            start = self.find_adjacent_object_node(start)
        start_index = self.node_index_mapping[start]
        target_node_index = self.node_index_mapping[target]
        queue.insert(0, target)
        predecessor_index = self.predecessors[start_index][target_node_index]
        if predecessor_index < 0:
            return [target]
        counter = 0
        while True:
            if predecessor_index < 0:
                return queue
            counter += 1
            queue.insert(0, self.index_node_mapping[predecessor_index])
            predecessor_index = self.predecessors[start_index][predecessor_index]
            if predecessor_index == start_index:
                break
        return queue

    def get_object_distance_matrix(self, only_free_agv: False):
        # print("\n###########")
        # print("NEW STEP")
        # print("############\n")
        # for agv in self.factory.agvs:
        #     print(agv.get_info())
        self.only_free_agv = only_free_agv
        self.construct_graph()
        # self.view_graph()
        dm = self.create_distance_matrix()
        # print(self.agv_paths)
        # self.draw_nx_graph()
        # print(dm)
        return dm

    def draw_nx_graph(self):
        csr_graph = csr_matrix(self.adjacency_matrix)
        G = nx.from_scipy_sparse_array(csr_graph,)
        pos_dict = {}
        for i,  coord in enumerate(self.agv_nodes.values()):
            pos_dict[i] = coord
        for i, coord in enumerate(self.nodes):
            pos_dict[i + len(self.agv_nodes)] = coord
        nx.draw(G, pos=pos_dict, with_labels=True)

# #
# if __name__ == "__main__":
#     factory = Factory()
#     # fac.create_temp_factory_machines_PAPER()
#     with open(
#             f"C:/Users/mente/PycharmProjects/ZellFTF_2DSim_PAPER_OR/VRP_Simulation/Random_Path_Factories/20241211_08-08-25_random_factory.pkl",
#             'rb') as inp:
#         factory = pickle.load(inp)
#         num_vehicles = 11
#         if num_vehicles >= 1:
#             factory.loading_stations.append(LoadingStation())
#             factory.loading_stations[6].pos_x = 31
#             factory.loading_stations[6].pos_y = 1
#             factory.loading_stations[6].length = 1
#             factory.loading_stations[6].width = 1
#             factory.agvs.append(AGV(start_position=[31, 1], static=True))
#             factory.agvs[6].name = 'AGV_6'
#             factory.agvs[6].factory = factory
#             factory.add_to_grid(factory.loading_stations[6])
#
#         if num_vehicles >= 2:
#             factory.loading_stations.append(LoadingStation())
#             factory.loading_stations[7].pos_x = 11
#             factory.loading_stations[7].pos_y = 17
#             factory.loading_stations[7].length = 1
#             factory.loading_stations[7].width = 1
#             factory.agvs.append(AGV(start_position=[11, 17], static=True))
#             factory.agvs[7].name = 'AGV_7'
#             factory.agvs[7].factory = factory
#             factory.add_to_grid(factory.loading_stations[7])
#
#         if num_vehicles >= 3:
#             factory.loading_stations.append(LoadingStation())
#             factory.loading_stations[8].pos_x = 19
#             factory.loading_stations[8].pos_y = 13
#             factory.loading_stations[8].length = 1
#             factory.loading_stations[8].width = 1
#             factory.agvs.append(AGV(start_position=[19, 13], static=True))
#             factory.agvs[8].name = 'AGV_8'
#             factory.agvs[8].factory = factory
#             factory.add_to_grid(factory.loading_stations[8])
#
#         if num_vehicles >= 4:
#             factory.loading_stations.append(LoadingStation())
#             factory.loading_stations[9].pos_x = 7
#             factory.loading_stations[9].pos_y = 13
#             factory.loading_stations[9].length = 1
#             factory.loading_stations[9].width = 1
#             factory.agvs.append(AGV(start_position=[7, 13], static=True))
#             factory.agvs[9].name = 'AGV_9'
#             factory.agvs[9].factory = factory
#             factory.add_to_grid(factory.loading_stations[9])
#
#         if num_vehicles >= 5:
#             factory.loading_stations.append(LoadingStation())
#             factory.loading_stations[10].pos_x = 23
#             factory.loading_stations[10].pos_y = 1
#             factory.loading_stations[10].length = 1
#             factory.loading_stations[10].width = 1
#             factory.agvs.append(AGV(start_position=[23, 1], static=True))
#             factory.agvs[10].name = 'AGV_10'
#             factory.agvs[10].factory = factory
#             factory.add_to_grid(factory.loading_stations[10])
#
#         if num_vehicles >= 6:
#             factory.loading_stations.append(LoadingStation())
#             factory.loading_stations[11].pos_x = 3
#             factory.loading_stations[11].pos_y = 9
#             factory.loading_stations[11].length = 1
#             factory.loading_stations[11].width = 1
#             factory.agvs.append(AGV(start_position=[3, 9], static=True))
#             factory.agvs[11].name = 'AGV_11'
#             factory.agvs[11].factory = factory
#             factory.add_to_grid(factory.loading_stations[11])
#
#         if num_vehicles >= 7:
#             factory.loading_stations.append(LoadingStation())
#             factory.loading_stations[12].pos_x = 31
#             factory.loading_stations[12].pos_y = 3
#             factory.loading_stations[12].length = 1
#             factory.loading_stations[12].width = 1
#             factory.agvs.append(AGV(start_position=[31, 3], static=True))
#             factory.agvs[12].name = 'AGV_12'
#             factory.agvs[12].factory = factory
#             factory.add_to_grid(factory.loading_stations[12])
#
#         if num_vehicles >= 8:
#             factory.loading_stations.append(LoadingStation())
#             factory.loading_stations[13].pos_x = 11
#             factory.loading_stations[13].pos_y = 19
#             factory.loading_stations[13].length = 1
#             factory.loading_stations[13].width = 1
#             factory.agvs.append(AGV(start_position=[11, 19], static=True))
#             factory.agvs[13].name = 'AGV_13'
#             factory.agvs[13].factory = factory
#             factory.add_to_grid(factory.loading_stations[13])
#
#         if num_vehicles >= 9:
#             factory.loading_stations.append(LoadingStation())
#             factory.loading_stations[14].pos_x = 19
#             factory.loading_stations[14].pos_y = 15
#             factory.loading_stations[14].length = 1
#             factory.loading_stations[14].width = 1
#             factory.agvs.append(AGV(start_position=[19, 15], static=True))
#             factory.agvs[14].name = 'AGV_14'
#             factory.agvs[14].factory = factory
#             factory.add_to_grid(factory.loading_stations[14])
#
#         if num_vehicles >= 10:
#             factory.loading_stations.append(LoadingStation())
#             factory.loading_stations[15].pos_x = 7
#             factory.loading_stations[15].pos_y = 15
#             factory.loading_stations[15].length = 1
#             factory.loading_stations[15].width = 1
#             factory.agvs.append(AGV(start_position=[7, 15], static=True))
#             factory.agvs[15].name = 'AGV_15'
#             factory.agvs[15].factory = factory
#             factory.add_to_grid(factory.loading_stations[15])
#
#         if num_vehicles >= 11:
#             factory.loading_stations.append(LoadingStation())
#             factory.loading_stations[16].pos_x = 23
#             factory.loading_stations[16].pos_y = 3
#             factory.loading_stations[16].length = 1
#             factory.loading_stations[16].width = 1
#             factory.agvs.append(AGV(start_position=[23, 3], static=True))
#             factory.agvs[16].name = 'AGV_16'
#             factory.agvs[16].factory = factory
#             factory.add_to_grid(factory.loading_stations[16])
#
#         if num_vehicles >= 12:
#             factory.loading_stations.append(LoadingStation())
#             factory.loading_stations[17].pos_x = 3
#             factory.loading_stations[17].pos_y = 11
#             factory.loading_stations[17].length = 1
#             factory.loading_stations[17].width = 1
#             factory.agvs.append(AGV(start_position=[3, 11], static=True))
#             factory.agvs[17].name = 'AGV_17'
#             factory.agvs[17].factory = factory
#             factory.add_to_grid(factory.loading_stations[17])
#     graph = FabricPathGraph(factory)
#     dm=graph.get_object_distance_matrix(only_free_agv=False)
#     # print(graph.get_predecessor_queue(fac.agvs[0], (1, 3)))
#     graph.view_graph()
#     # print(dm)


