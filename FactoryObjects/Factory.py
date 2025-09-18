# general packages
import config
import numpy as np
import json
import random

# own packages
from FactoryObjects.AGV import AGV
from FactoryObjects.LoadingStation import LoadingStation
from FactoryObjects.Machine import Machine
from FactoryObjects.Path import Path
from FactoryObjects.Warehouse import Warehouse
from FactoryObjects.Product import Product
from Sandbox.Factory_Path_Graph import FabricPathGraph


class Factory:
    def __init__(self):
        self.name = config.factory['name']
        self.length = config.factory['length']
        self.width = config.factory['width']
        self.cell_size = config.factory['cell_size']
        self.no_columns = int(self.length / self.cell_size)
        self.no_rows = int(self.width / self.cell_size)
        self.np_factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows))   # in this matrix all the data of
        # the factory cells is stored Zeilen, Spalten
        self.factory_grid_layout = self.np_factory_grid_layout.tolist()

        self.agvs = []
        self.loading_stations = []
        self.machines = []
        self.warehouses = []
        self.products = []
        self.product_types_list = []
        self.product_types = {'default_product_1': dict(length=500, width=500, weight=25.0)}
        self.create_default_product_types()
        self.products_id_count = 1000  # Product id's will start upwards with 1000

        self.path_graph = FabricPathGraph(factory=self)
        self.use_paths = False

        # self.create_temp_factory_machines()
        # self.create_temp_factory_machines_2()

        # print(f'Factory Grid Layout: {self.factory_grid_layout}')
        # print(f'Product Types: {self.product_types}')

    def create_default_product_types(self):
        self.product_types['default_product_2'] = dict(length=1000, width=1000, weight=100.0)
        self.product_types['default_product_3'] = dict(length=1500, width=1000, weight=150.0)
        self.product_types['default_product_4'] = dict(length=500, width=1000, weight=50.0)
        #print(self.product_types)
        self.dict_to_list()

    def clear_factory(self):
        """
        :return:
        """
        self.product_types = []
        self.products = []
        self.warehouses = []
        self.machines = []
        self.loading_stations = []
        self.agvs = []
        self.no_columns = int(self.length / self.cell_size)
        self.no_rows = int(self.width / self.cell_size)
        self.np_factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows))  # in this matrix all the data of
        # the factory cells is stored Zeilen, Spalten
        self.factory_grid_layout = self.np_factory_grid_layout.tolist()

        self.product_types = {'default_product_1': dict(length=500, width=500, weight=25.0)}
        self.create_default_product_types()
        self.products_id_count = 1000  # Product id's will start upwards with 1000

        pass

    def save_factory(self):
        pass

    def add_machine(self, factory_object):
        self.machines.append(factory_object)
        factory_object.factory = self

    def add_warehouse(self, factory_object):
        self.warehouses.append(factory_object)
        factory_object.factory = self

    def add_loading_station(self, factory_object):
        self.loading_stations.append(factory_object)
        factory_object.factory = self

    def add_agv(self, factory_object):
        self.agvs.append(factory_object)
        factory_object.factory = self
        print(self.agvs)
        print(factory_object.factory)

    def add_product_types(self, factory_object_list):
        self.product_types_list.append(factory_object_list)
        print(self.product_types_list)
        self.list_to_dict(self.product_types_list)

    def delete_machine(self, i):
        del self.machines[i]

    def delete_warehouse(self, i):
        del self.warehouses[i]

    def delete_loading_station(self, i):
        del self.loading_stations[i]

    def delete_agv(self, i):
        del self.agvs[i]

    def reload_settings(self):
        self.name = config.factory['name']
        self.length = config.factory['length']
        self.width = config.factory['width']
        self.cell_size = config.factory['cell_size']
        self.no_columns = int(self.length // self.cell_size)
        self.no_rows = int(self.width // self.cell_size)
        self.np_factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows))  # in this matrix all the data of
        # the factory cells is stored Zeilen, Spalten
        self.factory_grid_layout = self.np_factory_grid_layout.tolist()
        print(self.factory_grid_layout)

        # self.agv.reload_settings()
        # self.forklift.reload_settings()
        # self.machine.reload_settings()

    def reset(self):
        for agv in self.agvs:
            agv.reset()
        for machine in self.machines:
            machine.reset()
        for warehouse in self.warehouses:
            warehouse.reset()
        self.products = []

    def create_temp_factory_machines(self):
        self.length = 10
        self.width = 10
        self.no_columns = int(self.length // self.cell_size)
        self.no_rows = int(self.width // self.cell_size)
        self.factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows)).tolist()

        self.warehouses.append(Warehouse())
        self.warehouses[0].name = 'Warehouse_0'
        self.warehouses[0].pos_x = 0
        self.warehouses[0].pos_y = 8
        self.warehouses[0].length = 5
        self.warehouses[0].width = 2
        self.warehouses[0].pos_input = [4, 8]
        self.warehouses[0].pos_output = [1, 8]
        self.warehouses[0].input_products = ['four']    # ['four']
        self.warehouses[0].output_products = ['one']
        self.warehouses[0].factory = self
        self.warehouses[0].process_time = 5
        self.warehouses[0].rest_process_time = 5

        self.machines.append(Machine())
        self.machines[0].name = 'Maschine_0'
        self.machines[0].pos_x = 0
        self.machines[0].pos_y = 0
        self.machines[0].length = 3
        self.machines[0].width = 3
        self.machines[0].pos_input = [1, 2]
        self.machines[0].pos_output = [2, 1]
        self.machines[0].input_products = ['one']
        self.machines[0].output_products = ['two']
        self.machines[0].factory = self
        self.machines[0].process_time = 5
        self.machines[0].rest_process_time = 5
        self.machines[0].buffer_input = [1]

        self.machines.append(Machine())
        self.machines[1].name = 'Maschine_1'
        self.machines[1].pos_x = 7
        self.machines[1].pos_y = 0
        self.machines[1].length = 3
        self.machines[1].width = 3
        self.machines[1].pos_input = [7, 1]
        self.machines[1].pos_output = [8, 2]
        self.machines[1].input_products = ['two']
        self.machines[1].output_products = ['three']
        self.machines[1].factory = self
        self.machines[1].process_time = 5
        self.machines[1].rest_process_time = 5


        self.machines.append(Machine())
        self.machines[2].name = 'Maschine_2'
        self.machines[2].pos_x = 7
        self.machines[2].pos_y = 7
        self.machines[2].length = 3
        self.machines[2].width = 3
        self.machines[2].pos_input = [8, 7]
        self.machines[2].pos_output = [7, 8]
        self.machines[2].input_products = ['three']
        self.machines[2].output_products = ['four']
        self.machines[2].factory = self
        self.machines[2].process_time = 5
        self.machines[2].rest_process_time = 5

        self.agvs.append(AGV([0, 4]))
        self.agvs[0].name = 'AGV_0'
        self.agvs[0].factory = self

        self.agvs.append(AGV([0, 5]))
        self.agvs[1].name = 'AGV_1'
        self.agvs[1].factory = self

        self.agvs.append(AGV([0, 6]))
        self.agvs[2].name = 'AGV_2'
        self.agvs[2].factory = self

        self.agvs.append(AGV([0, 7]))
        self.agvs[3].name = 'AGV_3'
        self.agvs[3].factory = self

        self.agvs.append(AGV([5, 9]))
        self.agvs[4].name = 'AGV_4'
        self.agvs[4].factory = self

        self.agvs.append(AGV([6, 9]))
        self.agvs[5].name = 'AGV_5'
        self.agvs[5].factory = self

        self.loading_stations.append(LoadingStation())
        self.loading_stations[0].pos_x = 0
        self.loading_stations[0].pos_y = 7
        self.loading_stations[0].length = 1
        self.loading_stations[0].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[1].pos_x = 0
        self.loading_stations[1].pos_y = 6
        self.loading_stations[1].length = 1
        self.loading_stations[1].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[2].pos_x = 0
        self.loading_stations[2].pos_y = 5
        self.loading_stations[2].length = 1
        self.loading_stations[2].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[3].pos_x = 0
        self.loading_stations[3].pos_y = 4
        self.loading_stations[3].length = 1
        self.loading_stations[3].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[4].pos_x = 5
        self.loading_stations[4].pos_y = 9
        self.loading_stations[4].length = 1
        self.loading_stations[4].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[5].pos_x = 6
        self.loading_stations[5].pos_y = 9
        self.loading_stations[5].length = 1
        self.loading_stations[5].width = 1

        # self.loading_stations.append(LoadingStation())
        # self.loading_stations[6].pos_x = 3
        # self.loading_stations[6].pos_y = 0
        # self.loading_stations[6].length = 1
        # self.loading_stations[6].width = 1
        # self.agvs.append(AGV([3, 0]))
        # self.agvs[6].name = 'AGV_6'
        # self.agvs[6].factory = self
        #
        # self.loading_stations.append(LoadingStation())
        # self.loading_stations[7].pos_x = 4
        # self.loading_stations[7].pos_y = 0
        # self.loading_stations[7].length = 1
        # self.loading_stations[7].width = 1
        # self.agvs.append(AGV([4, 0]))
        # self.agvs[7].name = 'AGV_7'
        # self.agvs[7].factory = self
        #
        # self.loading_stations.append(LoadingStation())
        # self.loading_stations[8].pos_x = 5
        # self.loading_stations[8].pos_y = 0
        # self.loading_stations[8].length = 1
        # self.loading_stations[8].width = 1
        # self.agvs.append(AGV([5, 0]))
        # self.agvs[8].name = 'AGV_8'
        # self.agvs[8].factory = self
        #
        # self.loading_stations.append(LoadingStation())
        # self.loading_stations[9].pos_x = 6
        # self.loading_stations[9].pos_y = 0
        # self.loading_stations[9].length = 1
        # self.loading_stations[9].width = 1
        # self.agvs.append(AGV([6, 0]))
        # self.agvs[9].name = 'AGV_9'
        # self.agvs[9].factory = self

        self.fill_grid()

        self.factory_grid_layout[3][1] = Path()
        self.factory_grid_layout[4][1] = Path()
        self.factory_grid_layout[5][1] = Path()
        self.factory_grid_layout[6][1] = Path()
        self.factory_grid_layout[6][2] = Path()
        self.factory_grid_layout[6][3] = Path()
        self.factory_grid_layout[7][3] = Path()
        self.factory_grid_layout[8][3] = Path()
        self.factory_grid_layout[8][4] = Path()
        self.factory_grid_layout[8][5] = Path()
        self.factory_grid_layout[8][6] = Path()
        self.factory_grid_layout[7][6] = Path()
        self.factory_grid_layout[6][6] = Path()
        self.factory_grid_layout[6][7] = Path()
        self.factory_grid_layout[6][8] = Path()
        self.factory_grid_layout[5][8] = Path()
        self.factory_grid_layout[5][7] = Path()
        self.factory_grid_layout[4][7] = Path()
        self.factory_grid_layout[3][7] = Path()
        self.factory_grid_layout[2][7] = Path()
        self.factory_grid_layout[1][7] = Path()
        self.factory_grid_layout[1][6] = Path()
        self.factory_grid_layout[1][5] = Path()
        self.factory_grid_layout[1][4] = Path()
        self.factory_grid_layout[1][3] = Path()
        self.factory_grid_layout[2][3] = Path()
        self.factory_grid_layout[3][3] = Path()
        self.factory_grid_layout[3][2] = Path()



        print(self.factory_grid_layout)

        self.product_types['one'] = dict(length=1100, width=600, weight=4.5)  # dict(length=1100, width=600, weight=9.0)
        self.product_types['two'] = dict(length=600, width=600, weight=4.5)  # dict(length=600, width=600, weight=4.5)
        self.product_types['three'] = dict(length=250, width=250, weight=4.5)
        self.product_types['four'] = dict(length=250, width=250, weight=4.5)
        print(self.product_types)
        print(self.machines)

        # self.factory_grid_layout[5][5] = Path()

    def create_temp_factory_machines_deadlock(self):
        self.length = 10
        self.width = 10
        self.no_columns = int(self.length // self.cell_size)
        self.no_rows = int(self.width // self.cell_size)
        self.factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows)).tolist()

        self.warehouses.append(Warehouse())
        self.warehouses[0].name = 'Warehouse_0'
        self.warehouses[0].pos_x = 0
        self.warehouses[0].pos_y = 8
        self.warehouses[0].length = 5
        self.warehouses[0].width = 2
        self.warehouses[0].pos_input = [4, 8]
        self.warehouses[0].pos_output = [1, 8]
        self.warehouses[0].input_products = ['four']    # ['four']
        self.warehouses[0].output_products = ['one']
        self.warehouses[0].factory = self
        self.warehouses[0].process_time = 5
        self.warehouses[0].rest_process_time = 5

        self.machines.append(Machine())
        self.machines[0].name = 'Maschine_0'
        self.machines[0].pos_x = 0
        self.machines[0].pos_y = 0
        self.machines[0].length = 3
        self.machines[0].width = 3
        self.machines[0].pos_input = [1, 2]
        self.machines[0].pos_output = [2, 1]
        self.machines[0].input_products = ['one']
        self.machines[0].output_products = ['two']
        self.machines[0].factory = self
        self.machines[0].process_time = 5
        self.machines[0].rest_process_time = 5

        self.machines.append(Machine())
        self.machines[1].name = 'Maschine_1'
        self.machines[1].pos_x = 7
        self.machines[1].pos_y = 0
        self.machines[1].length = 3
        self.machines[1].width = 3
        self.machines[1].pos_input = [7, 1]
        self.machines[1].pos_output = [8, 2]
        self.machines[1].input_products = ['two']
        self.machines[1].output_products = ['three']
        self.machines[1].factory = self
        self.machines[1].process_time = 5
        self.machines[1].rest_process_time = 5


        self.machines.append(Machine())
        self.machines[2].name = 'Maschine_2'
        self.machines[2].pos_x = 7
        self.machines[2].pos_y = 7
        self.machines[2].length = 3
        self.machines[2].width = 3
        self.machines[2].pos_input = [8, 7]
        self.machines[2].pos_output = [7, 8]
        self.machines[2].input_products = ['three']
        self.machines[2].output_products = ['four']
        self.machines[2].factory = self
        self.machines[2].process_time = 5
        self.machines[2].rest_process_time = 5

        self.agvs.append(AGV([1, 2]))
        self.agvs[0].name = 'AGV_0'
        self.agvs[0].factory = self

        self.agvs.append(AGV([1.5, 2]))
        self.agvs[1].name = 'AGV_1'
        self.agvs[1].factory = self

        self.agvs.append(AGV([1, 2.5]))
        self.agvs[2].name = 'AGV_2'
        self.agvs[2].factory = self

        self.agvs.append(AGV([1.5, 2.5]))
        self.agvs[3].name = 'AGV_3'
        self.agvs[3].factory = self

        self.agvs.append(AGV([1, 3]))
        self.agvs[4].name = 'AGV_4'
        self.agvs[4].factory = self

        self.agvs.append(AGV([1.5, 3]))
        self.agvs[5].name = 'AGV_5'
        self.agvs[5].factory = self

        self.loading_stations.append(LoadingStation())
        self.loading_stations[0].pos_x = 0
        self.loading_stations[0].pos_y = 7
        self.loading_stations[0].length = 1
        self.loading_stations[0].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[1].pos_x = 0
        self.loading_stations[1].pos_y = 6
        self.loading_stations[1].length = 1
        self.loading_stations[1].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[2].pos_x = 0
        self.loading_stations[2].pos_y = 5
        self.loading_stations[2].length = 1
        self.loading_stations[2].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[3].pos_x = 0
        self.loading_stations[3].pos_y = 4
        self.loading_stations[3].length = 1
        self.loading_stations[3].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[4].pos_x = 5
        self.loading_stations[4].pos_y = 9
        self.loading_stations[4].length = 1
        self.loading_stations[4].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[5].pos_x = 6
        self.loading_stations[5].pos_y = 9
        self.loading_stations[5].length = 1
        self.loading_stations[5].width = 1

        # self.loading_stations.append(LoadingStation())
        # self.loading_stations[6].pos_x = 3
        # self.loading_stations[6].pos_y = 0
        # self.loading_stations[6].length = 1
        # self.loading_stations[6].width = 1
        # self.agvs.append(AGV([3, 0]))
        # self.agvs[6].name = 'AGV_6'
        # self.agvs[6].factory = self
        #
        # self.loading_stations.append(LoadingStation())
        # self.loading_stations[7].pos_x = 4
        # self.loading_stations[7].pos_y = 0
        # self.loading_stations[7].length = 1
        # self.loading_stations[7].width = 1
        # self.agvs.append(AGV([4, 0]))
        # self.agvs[7].name = 'AGV_7'
        # self.agvs[7].factory = self
        #
        # self.loading_stations.append(LoadingStation())
        # self.loading_stations[8].pos_x = 5
        # self.loading_stations[8].pos_y = 0
        # self.loading_stations[8].length = 1
        # self.loading_stations[8].width = 1
        # self.agvs.append(AGV([5, 0]))
        # self.agvs[8].name = 'AGV_8'
        # self.agvs[8].factory = self
        #
        # self.loading_stations.append(LoadingStation())
        # self.loading_stations[9].pos_x = 6
        # self.loading_stations[9].pos_y = 0
        # self.loading_stations[9].length = 1
        # self.loading_stations[9].width = 1
        # self.agvs.append(AGV([6, 0]))
        # self.agvs[9].name = 'AGV_9'
        # self.agvs[9].factory = self

        self.fill_grid()
        print(self.factory_grid_layout)

        self.product_types['one'] = dict(length=1100, width=600, weight=4.5)  # dict(length=1100, width=600, weight=9.0)
        self.product_types['two'] = dict(length=600, width=600, weight=4.5)  # dict(length=600, width=600, weight=4.5)
        self.product_types['three'] = dict(length=250, width=250, weight=4.5)
        self.product_types['four'] = dict(length=250, width=250, weight=4.5)
        print(self.product_types)
        print(self.machines)

        # self.factory_grid_layout[5][5] = Path()

    def create_temp_factory_machines_2(self):
        self.length = 15
        self.width = 15
        self.no_columns = int(self.length // self.cell_size)
        self.no_rows = int(self.width // self.cell_size)
        self.factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows)).tolist()

        self.warehouses.append(Warehouse())
        self.warehouses[0].pos_x = 0
        self.warehouses[0].pos_y = 13
        self.warehouses[0].length = 5
        self.warehouses[0].width = 2
        self.warehouses[0].pos_input = [4, 13]
        self.warehouses[0].pos_output = [1, 13]
        self.warehouses[0].input_products = ['default_product_1']
        self.warehouses[0].output_products = ['default_product_3']
        self.warehouses[0].factory = self

        self.machines.append(Machine())
        self.machines[0].pos_x = 0
        self.machines[0].pos_y = 0
        self.machines[0].length = 5
        self.machines[0].width = 3
        self.machines[0].pos_input = [1, 2]
        self.machines[0].pos_output = [4, 1]
        self.machines[0].input_products = ['default_product_3']
        self.machines[0].output_products = ['default_product_2']
        self.machines[0].factory = self

        self.machines.append(Machine())
        self.machines[1].pos_x = 10
        self.machines[1].pos_y = 0
        self.machines[1].length = 5
        self.machines[1].width = 5
        self.machines[1].pos_input = [10, 2]
        self.machines[1].pos_output = [13, 4]
        self.machines[1].input_products = ['default_product_2']
        self.machines[1].output_products = ['default_product_1']
        self.machines[1].factory = self
        self.machines[1].process_time = 10
        self.machines[1].rest_process_time = 10
        '''
        self.machines.append(Machine())
        self.machines[2].pos_x = 7
        self.machines[2].pos_y = 7
        self.machines[2].length = 3
        self.machines[2].width = 3
        self.machines[2].pos_input = [8, 7]
        self.machines[2].pos_output = [7, 8]
        self.machines[2].input_products = ['three']
        self.machines[2].output_products = ['four']
        self.machines[2].factory = self
        '''
        self.agvs.append(AGV([14, 14]))
        self.agvs[0].factory = self

        self.agvs.append(AGV([13, 14]))
        self.agvs[1].factory = self

        self.agvs.append(AGV([12, 14]))
        self.agvs[2].factory = self

        self.agvs.append(AGV([11, 14]))
        self.agvs[3].factory = self

        self.agvs.append(AGV([10, 14]))
        self.agvs[4].factory = self

        self.agvs.append(AGV([9, 14]))
        self.agvs[5].factory = self

        self.agvs.append(AGV([8, 14]))
        self.agvs[6].factory = self

        self.agvs.append(AGV([7, 14]))
        self.agvs[7].factory = self


        self.loading_stations.append(LoadingStation())
        self.loading_stations[0].pos_x = 14
        self.loading_stations[0].pos_y = 14
        self.loading_stations[0].length = 1
        self.loading_stations[0].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[1].pos_x = 13
        self.loading_stations[1].pos_y = 14
        self.loading_stations[1].length = 1
        self.loading_stations[1].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[2].pos_x = 12
        self.loading_stations[2].pos_y = 14
        self.loading_stations[2].length = 1
        self.loading_stations[2].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[3].pos_x = 11
        self.loading_stations[3].pos_y = 14
        self.loading_stations[3].length = 1
        self.loading_stations[3].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[4].pos_x = 10
        self.loading_stations[4].pos_y = 14
        self.loading_stations[4].length = 1
        self.loading_stations[4].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[5].pos_x = 9
        self.loading_stations[5].pos_y = 14
        self.loading_stations[5].length = 1
        self.loading_stations[5].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[6].pos_x = 8
        self.loading_stations[6].pos_y = 14
        self.loading_stations[6].length = 1
        self.loading_stations[6].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[7].pos_x = 7
        self.loading_stations[7].pos_y = 14
        self.loading_stations[7].length = 1
        self.loading_stations[7].width = 1

        self.fill_grid()


    def create_temp_factory_machines_3(self):
        self.length = 12
        self.width = 9
        self.no_columns = int(self.length // self.cell_size)
        self.no_rows = int(self.width // self.cell_size)
        self.factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows)).tolist()

        self.warehouses.append(Warehouse())
        self.warehouses[0].name = "Warehouse_1"
        self.warehouses[0].pos_x = 0
        self.warehouses[0].pos_y = 7
        self.warehouses[0].length = 5
        self.warehouses[0].width = 2
        self.warehouses[0].pos_input = [4, 7]
        self.warehouses[0].pos_output = [1, 7]
        self.warehouses[0].input_products = ['four']    # ['four']
        self.warehouses[0].output_products = ['one']
        self.warehouses[0].factory = self
        self.warehouses[0].process_time = 5
        self.warehouses[0].rest_process_time = 5

        self.machines.append(Machine())
        self.machines[0].name = "Machine_0"
        self.machines[0].pos_x = 0
        self.machines[0].pos_y = 0
        self.machines[0].length = 3
        self.machines[0].width = 3
        self.machines[0].pos_input = [1, 2]
        self.machines[0].pos_output = [2, 1]
        self.machines[0].input_products = ['one']
        self.machines[0].output_products = ['two']
        self.machines[0].buffer_input = [10]
        self.machines[0].process_time = 5
        self.machines[0].rest_process_time = 5
        self.machines[0].factory = self

        self.machines.append(Machine())
        self.machines[1].name = "Machine_1"
        self.machines[1].pos_x = 9
        self.machines[1].pos_y = 0
        self.machines[1].length = 3
        self.machines[1].width = 3
        self.machines[1].pos_input = [9, 1]
        self.machines[1].pos_output = [10, 2]
        self.machines[1].input_products = ['two']
        self.machines[1].output_products = ['three']
        self.machines[1].factory = self
        self.machines[1].process_time = 5
        self.machines[1].rest_process_time = 5


        self.machines.append(Machine())
        self.machines[2].name = "Machine_2"
        self.machines[2].pos_x = 9
        self.machines[2].pos_y = 6
        self.machines[2].length = 3
        self.machines[2].width = 3
        self.machines[2].pos_input = [10, 6]
        self.machines[2].pos_output = [9, 7]
        self.machines[2].input_products = ['three']
        self.machines[2].output_products = ['four']
        self.machines[2].factory = self
        self.machines[2].process_time = 5
        self.machines[2].rest_process_time = 5

        self.agvs.append(AGV([0, 3]))
        self.agvs[0].name = "0"
        self.agvs[0].factory = self

        self.agvs.append(AGV([0, 4]))
        self.agvs[1].name = "1"
        self.agvs[1].factory = self

        self.agvs.append(AGV([0, 5]))
        self.agvs[2].name = "2"
        self.agvs[2].factory = self

        self.agvs.append(AGV([0, 6]))
        self.agvs[3].name = "3"
        self.agvs[3].factory = self

        self.agvs.append(AGV([3, 0]))
        self.agvs[4].name = "4"
        self.agvs[4].factory = self

        self.agvs.append(AGV([4, 0]))
        self.agvs[5].name = "5"
        self.agvs[5].factory = self

        # self.agvs.append(AGV([5, 0]))
        # self.agvs[6].name = "6"
        # self.agvs[6].factory = self
        #
        # self.agvs.append(AGV([6, 0]))
        # self.agvs[7].name = "7"
        # self.agvs[7].factory = self
        #
        # self.agvs.append(AGV([7, 0]))
        # self.agvs[8].name = "8"
        # self.agvs[8].factory = self
        #
        # self.agvs.append(AGV([8, 0]))
        # self.agvs[9].name = "9"
        # self.agvs[9].factory = self

        self.loading_stations.append(LoadingStation())
        self.loading_stations[0].pos_x = 0
        self.loading_stations[0].pos_y = 6
        self.loading_stations[0].length = 1
        self.loading_stations[0].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[1].pos_x = 0
        self.loading_stations[1].pos_y = 5
        self.loading_stations[1].length = 1
        self.loading_stations[1].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[2].pos_x = 0
        self.loading_stations[2].pos_y = 4
        self.loading_stations[2].length = 1
        self.loading_stations[2].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[3].pos_x = 0
        self.loading_stations[3].pos_y = 3
        self.loading_stations[3].length = 1
        self.loading_stations[3].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[4].pos_x = 3
        self.loading_stations[4].pos_y = 0
        self.loading_stations[4].length = 1
        self.loading_stations[4].width = 1

        self.loading_stations.append(LoadingStation())
        self.loading_stations[5].pos_x = 4
        self.loading_stations[5].pos_y = 0
        self.loading_stations[5].length = 1
        self.loading_stations[5].width = 1

        # self.loading_stations.append(LoadingStation())
        # self.loading_stations[6].pos_x = 5
        # self.loading_stations[6].pos_y = 0
        # self.loading_stations[6].length = 1
        # self.loading_stations[6].width = 1
        #
        # self.loading_stations.append(LoadingStation())
        # self.loading_stations[7].pos_x = 6
        # self.loading_stations[7].pos_y = 0
        # self.loading_stations[7].length = 1
        # self.loading_stations[7].width = 1
        #
        # self.loading_stations.append(LoadingStation())
        # self.loading_stations[8].pos_x = 7
        # self.loading_stations[8].pos_y = 0
        # self.loading_stations[8].length = 1
        # self.loading_stations[8].width = 1
        #
        # self.loading_stations.append(LoadingStation())
        # self.loading_stations[9].pos_x = 8
        # self.loading_stations[9].pos_y = 0
        # self.loading_stations[9].length = 1
        # self.loading_stations[9].width = 1

        self.fill_grid()

        # self.factory_grid_layout[5][5] = Block()

        self.factory_grid_layout[1][3] = Path()
        self.factory_grid_layout[1][4] = Path()
        self.factory_grid_layout[1][5] = Path()
        self.factory_grid_layout[1][6] = Path()

        self.factory_grid_layout[2][3] = Path()
        self.factory_grid_layout[3][3] = Path()
        self.factory_grid_layout[3][2] = Path()

        self.factory_grid_layout[3][1] = Path()
        self.factory_grid_layout[4][1] = Path()
        self.factory_grid_layout[5][1] = Path()
        self.factory_grid_layout[6][1] = Path()
        self.factory_grid_layout[7][1] = Path()
        self.factory_grid_layout[8][1] = Path()

        self.factory_grid_layout[8][2] = Path()
        self.factory_grid_layout[8][3] = Path()
        self.factory_grid_layout[9][3] = Path()

        self.factory_grid_layout[10][3] = Path()
        self.factory_grid_layout[10][4] = Path()
        self.factory_grid_layout[10][5] = Path()

        self.factory_grid_layout[9][5] = Path()
        self.factory_grid_layout[8][5] = Path()
        self.factory_grid_layout[8][6] = Path()

        self.factory_grid_layout[5][7] = Path()
        self.factory_grid_layout[6][7] = Path()
        self.factory_grid_layout[7][7] = Path()
        self.factory_grid_layout[8][7] = Path()

        self.factory_grid_layout[3][4] = Path()
        self.factory_grid_layout[3][5] = Path()

        self.factory_grid_layout[5][6] = Path()
        self.factory_grid_layout[4][6] = Path()
        self.factory_grid_layout[3][6] = Path()
        self.factory_grid_layout[2][6] = Path()

        # print(self.factory_grid_layout)

        self.product_types['one'] = dict(length=1100, width=600, weight=4.5)  # dict(length=1100, width=600, weight=9.0)
        self.product_types['two'] = dict(length=600, width=600, weight=4.5)  # dict(length=600, width=600, weight=4.5)
        self.product_types['three'] = dict(length=250, width=250, weight=4.5)
        self.product_types['four'] = dict(length=250, width=250, weight=4.5)
        # print(self.product_types)
        # print(self.machines)

    def create_temp_factory_machines_PAPER(self):
        self.length = 24
        self.width = 14
        self.no_columns = int(self.length // self.cell_size)
        self.no_rows = int(self.width // self.cell_size)
        self.factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows)).tolist()

        self.product_types['W0_Output'] = dict(length=800, width=800, weight=5)
        self.product_types['W1_Output'] = dict(length=800, width=800, weight=5)
        self.product_types['M0_Output'] = dict(length=800, width=800, weight=5)
        self.product_types['M1_Output'] = dict(length=1200, width=800, weight=5)
        self.product_types['M2_Output'] = dict(length=250, width=250, weight=5)
        self.product_types['M3_Output'] = dict(length=250, width=250, weight=5)
        self.product_types['M4_Output'] = dict(length=1200, width=800, weight=5)
        self.product_types['no_input'] = dict(length=250, width=250, weight=5)
        self.product_types['no_output'] = dict(length=250, width=250, weight=5)


        self.warehouses.append(Warehouse())
        self.warehouses[0].name = 'W0'
        self.warehouses[0].pos_x = 0
        self.warehouses[0].pos_y = 0
        self.warehouses[0].length = 4
        self.warehouses[0].width = 4
        self.warehouses[0].pos_input = [1, 3]
        self.warehouses[0].pos_output = [3, 1]
        self.warehouses[0].input_products = ['no_input']
        self.warehouses[0].output_products = ['W0_Output']
        self.warehouses[0].process_time = 15
        self.warehouses[0].rest_process_time = 15
        self.warehouses[0].factory = self

        # initial_product = self.create_product('W0_Output')
        # initial_product.stored_in = self.warehouses[0]
        # self.warehouses[0].buffer_output_load.append(initial_product)

        self.warehouses.append(Warehouse())
        self.warehouses[1].name = 'W1'
        self.warehouses[1].pos_x = 0
        self.warehouses[1].pos_y = 5
        self.warehouses[1].length = 4
        self.warehouses[1].width = 4
        self.warehouses[1].pos_input = [1, 5]
        self.warehouses[1].pos_output = [3, 6]
        self.warehouses[1].input_products = ['no_input']
        self.warehouses[1].output_products = ['W1_Output']
        self.warehouses[1].process_time = 35
        self.warehouses[1].rest_process_time = 35
        self.warehouses[1].factory = self

        # initial_product = self.create_product('W1_Output')
        # initial_product.stored_in = self.warehouses[1]
        # self.warehouses[1].buffer_output_load.append(initial_product)

        self.warehouses.append(Warehouse())
        self.warehouses[2].name = 'W2'
        self.warehouses[2].pos_x = 0
        self.warehouses[2].pos_y = 10
        self.warehouses[2].length = 4
        self.warehouses[2].width = 2
        self.warehouses[2].pos_input = [3, 11]
        self.warehouses[2].pos_output = [1, 10]
        self.warehouses[2].input_products = ['M4_Output']
        self.warehouses[2].output_products = ['no_output']
        self.warehouses[2].process_time = 1000000000
        self.warehouses[2].rest_process_time = 1000000000
        self.warehouses[2].factory = self

        self.warehouses.append(Warehouse())
        self.warehouses[3].name = 'W3'
        self.warehouses[3].pos_x = 0
        self.warehouses[3].pos_y = 12
        self.warehouses[3].length = 4
        self.warehouses[3].width = 2
        self.warehouses[3].pos_input = [3, 12]
        self.warehouses[3].pos_output = [3, 13]
        self.warehouses[3].input_products = ['M3_Output']
        self.warehouses[3].output_products = ['no_output']
        self.warehouses[3].process_time = 1000000000
        self.warehouses[3].rest_process_time = 1000000000
        self.warehouses[3].factory = self

        self.machines.append(Machine())
        self.machines[0].name = 'M0'
        self.machines[0].pos_x = 10
        self.machines[0].pos_y = 0
        self.machines[0].length = 4
        self.machines[0].width = 4
        self.machines[0].pos_input = [10, 3]
        self.machines[0].pos_output = [13, 3]
        self.machines[0].input_products = ['W0_Output']
        self.machines[0].output_products = ['M0_Output']
        self.machines[0].process_time = 15
        self.machines[0].rest_process_time = 15
        self.machines[0].buffer_input = [1]
        self.machines[0].buffer_output = [1]
        self.machines[0].amount_of_resulting_products = 1
        self.machines[0].factory = self

        self.machines.append(Machine())
        self.machines[1].name = 'M1'
        self.machines[1].pos_x = 20
        self.machines[1].pos_y = 0
        self.machines[1].length = 4
        self.machines[1].width = 4
        self.machines[1].pos_input = [20, 1]
        self.machines[1].pos_output = [23, 3]
        self.machines[1].input_products = ['M0_Output']
        self.machines[1].output_products = ['M1_Output']
        self.machines[1].process_time = 15
        self.machines[1].rest_process_time = 15
        self.machines[1].buffer_input = [1]
        self.machines[1].buffer_output = [1]
        self.machines[1].amount_of_resulting_products = 1
        self.machines[1].factory = self

        self.machines.append(Machine())
        self.machines[2].name = 'M2'
        self.machines[2].pos_x = 20
        self.machines[2].pos_y = 10
        self.machines[2].length = 4
        self.machines[2].width = 4
        self.machines[2].pos_input = [23, 10]
        self.machines[2].pos_output = [20, 13]
        self.machines[2].input_products = ['M1_Output']
        self.machines[2].output_products = ['M2_Output']
        self.machines[2].process_time = 15
        self.machines[2].rest_process_time = 15
        self.machines[2].buffer_input = [1]
        self.machines[2].buffer_output = [10]
        self.machines[2].amount_of_resulting_products = 6
        self.machines[2].factory = self

        self.machines.append(Machine())
        self.machines[3].name = 'M3'
        self.machines[3].pos_x = 15
        self.machines[3].pos_y = 5
        self.machines[3].length = 4
        self.machines[3].width = 4
        self.machines[3].pos_input = [15, 5]
        self.machines[3].pos_output = [18, 5]
        self.machines[3].input_products = ['W1_Output']
        self.machines[3].output_products = ['M3_Output']
        self.machines[3].process_time = 15
        self.machines[3].rest_process_time = 15
        self.machines[3].buffer_input = [1]
        self.machines[3].buffer_output = [1]
        self.machines[3].amount_of_resulting_products = 1
        self.machines[3].factory = self

        self.machines.append(Machine())
        self.machines[4].name = 'M4'
        self.machines[4].pos_x = 5
        self.machines[4].pos_y = 9
        self.machines[4].length = 9
        self.machines[4].width = 4
        self.machines[4].pos_input = [13, 12]
        self.machines[4].pos_output = [10, 12]
        self.machines[4].input_products = ['M2_Output']
        self.machines[4].output_products = ['M4_Output']
        self.machines[4].process_time = 15
        self.machines[4].rest_process_time = 15
        self.machines[4].buffer_input = [20]
        self.machines[4].buffer_output = [1]
        self.machines[4].amount_of_resulting_products = 1
        self.machines[4].factory = self


        STARTING_POINT = 4
        for i in range(6):
            self.loading_stations.append(LoadingStation())
            self.loading_stations[i].name = f'Loading_Station_{i}'
            self.loading_stations[i].pos_x = STARTING_POINT + i
            self.loading_stations[i].pos_y = 0
            self.loading_stations[i].length = 1
            self.loading_stations[i].width = 1
            self.agvs.append(AGV([STARTING_POINT+i, 0]))
            self.agvs[i].name = f"AGV_{i}"
            self.agvs[i].factory = self

        for i in range(6):
            self.loading_stations.append(LoadingStation())
            self.loading_stations[i+6].name = f'Loading_Station_{i+6}'
            self.loading_stations[i+6].pos_x = STARTING_POINT + 10 + i
            self.loading_stations[i+6].pos_y = 0
            self.loading_stations[i+6].length = 1
            self.loading_stations[i+6].width = 1
            self.agvs.append(AGV([STARTING_POINT+i+10, 0]))
            self.agvs[i+6].name = f"AGV_{i+6}"
            self.agvs[i+6].factory = self

        self.fill_grid()

        self.factory_grid_layout[9][2] = Path()
        self.factory_grid_layout[9][3] = Path()
        for x in range(6):
            self.factory_grid_layout[x+STARTING_POINT][1] = Path()
        for x in range(24):
            self.factory_grid_layout[x+0][4] = Path()
        for x in range(15):
            self.factory_grid_layout[x+5][13] = Path()
        for x in range(4):
            self.factory_grid_layout[x+20][9] = Path()
        for x in range(6):
            self.factory_grid_layout[x+14][1] = Path()
        for x in range(4):
            self.factory_grid_layout[x][9] = Path()
        for y in range(12):
            self.factory_grid_layout[4][y+2] = Path()
        for y in range(4):
            self.factory_grid_layout[23][y+5] = Path()
        for y in range(4):
            self.factory_grid_layout[19][y+9] = Path()
        for y in range(2):
            self.factory_grid_layout[14][y+2] = Path()
            self.factory_grid_layout[19][y+2] = Path()

    def create_temp_factory_machines_4(self):
        self.length = 15
        self.width = 20
        self.no_columns = int(self.length // self.cell_size)
        self.no_rows = int(self.width // self.cell_size)
        self.factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows)).tolist()

        self.product_types['lager_m0'] = dict(length=500, width=500, weight=1)
        self.product_types['m0_m1'] = dict(length=500, width=500, weight=1)
        self.product_types['m1_m2'] = dict(length=500, width=500, weight=1)
        self.product_types['m2_lager'] = dict(length=500, width=500, weight=1)
        self.product_types['lager_m3'] = dict(length=500, width=500, weight=1)
        self.product_types['m3_m4'] = dict(length=500, width=500, weight=1)
        self.product_types['m4_m5'] = dict(length=500, width=500, weight=1)
        self.product_types['m5_lager'] = dict(length=500, width=500, weight=1)
        self.product_types['empty'] = dict(length=500, width=500, weight=1)
        self.product_types['empty_w0'] = dict(length=500, width=500, weight=1)

        self.warehouses.append(Warehouse())
        self.warehouses[0].name = 'W0'
        self.warehouses[0].pos_x = 9
        self.warehouses[0].pos_y = 6
        self.warehouses[0].length = 2
        self.warehouses[0].width = 2
        self.warehouses[0].pos_input = [10, 6]
        self.warehouses[0].pos_output = [9, 6]
        self.warehouses[0].input_products = ['m2_lager']
        self.warehouses[0].output_products = ['lager_m0']
        self.warehouses[0].factory = self

        self.warehouses.append(Warehouse())
        self.warehouses[1].name = 'W1'
        self.warehouses[1].pos_x = 8
        self.warehouses[1].pos_y = 12
        self.warehouses[1].length = 2
        self.warehouses[1].width = 2
        self.warehouses[1].pos_input = [9, 13]
        self.warehouses[1].pos_output = [9, 12]
        self.warehouses[1].input_products = ['m5_lager']
        self.warehouses[1].output_products = ['lager_m3']
        self.warehouses[1].factory = self

        self.machines.append(Machine())
        self.machines[0].name = 'M0'
        self.machines[0].pos_x = 0
        self.machines[0].pos_y = 3
        self.machines[0].length = 2
        self.machines[0].width = 2
        self.machines[0].pos_input = [1, 4]
        self.machines[0].pos_output = [1, 3]
        self.machines[0].input_products = ['lager_m0']
        self.machines[0].output_products = ['m0_m1']
        self.machines[0].factory = self

        self.machines.append(Machine())
        self.machines[1].name = 'M1'
        self.machines[1].pos_x = 7
        self.machines[1].pos_y = 0
        self.machines[1].length = 2
        self.machines[1].width = 2
        self.machines[1].pos_input = [7, 1]
        self.machines[1].pos_output = [8, 1]
        self.machines[1].input_products = ['m0_m1']
        self.machines[1].output_products = ['m1_m2']
        self.machines[1].factory = self

        self.machines.append(Machine())
        self.machines[2].name = 'M2'
        self.machines[2].pos_x = 13
        self.machines[2].pos_y = 3
        self.machines[2].length = 2
        self.machines[2].width = 2
        self.machines[2].pos_input = [13, 3]
        self.machines[2].pos_output = [13, 4]
        self.machines[2].input_products = ['m1_m2']
        self.machines[2].output_products = ['m2_lager']
        self.machines[2].factory = self

        self.machines.append(Machine())
        self.machines[3].name = 'M3'
        self.machines[3].pos_x = 12
        self.machines[3].pos_y = 14
        self.machines[3].length = 2
        self.machines[3].width = 2
        self.machines[3].pos_input = [12, 14]
        self.machines[3].pos_output = [12, 15]
        self.machines[3].input_products = ['lager_m3']
        self.machines[3].output_products = ['m3_m4']
        self.machines[3].factory = self

        self.machines.append(Machine())
        self.machines[4].name = 'M4'
        self.machines[4].pos_x = 10
        self.machines[4].pos_y = 18
        self.machines[4].length = 2
        self.machines[4].width = 2
        self.machines[4].pos_input = [11, 18]
        self.machines[4].pos_output = [10, 18]
        self.machines[4].input_products = ['m3_m4']
        self.machines[4].output_products = ['m4_m5']
        self.machines[4].factory = self

        self.machines.append(Machine())
        self.machines[5].name = 'M5'
        self.machines[5].pos_x = 3
        self.machines[5].pos_y = 16
        self.machines[5].length = 2
        self.machines[5].width = 2
        self.machines[5].pos_input = [4, 17]
        self.machines[5].pos_output = [4, 16]
        self.machines[5].input_products = ['m4_m5']
        self.machines[5].output_products = ['m5_lager']
        self.machines[5].factory = self

        self.agvs.append(AGV([8, 5]))
        self.agvs[0].name = 0
        self.agvs[0].factory = self

        self.agvs.append(AGV([9, 10]))
        self.agvs[1].name = 1
        self.agvs[1].factory = self

        self.fill_grid()


    def create_temp_factory_machines_5(self):
        '''
        This function creates a default factory, where 3 machines get deliveries from a warehouse and between each other
        W-M1    - 1 AGV
        W-M2    - 6 AGV
        W-M3    - 4 AGV
        M1-M3   - 6 AGV
        M3-W    - 6 AGV
        M3-M2   - 4 AGV
        ________
        |     1|
        |      |
        |      |
        |D W  2|
        |      |
        |      |
        |      |
        |      |
        |     3|
        :param self:
        :return:
        '''
        print('############################################')
        print('##### Creating default factory for VRP #####')
        print('############################################')
        self.length = 10
        self.width = 10
        self.no_columns = int(self.length // self.cell_size)
        self.no_rows = int(self.width // self.cell_size)
        self.factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows)).tolist()

        self.product_types['lager_m1'] = dict(length=500, width=500, weight=1)
        self.product_types['lager_m2'] = dict(length=1500, width=1000, weight=4.5)
        self.product_types['lager_m3'] = dict(length=1000, width=1000, weight=4.5)
        self.product_types['m1_m3'] = dict(length=1200, width=1000, weight=4.5)
        self.product_types['m3_lager'] = dict(length=1500, width=800, weight=4.5)
        self.product_types['m3_m2'] = dict(length=1000, width=800, weight=4.5)

        print(self.product_types)

        self.warehouses.append(Warehouse())
        self.warehouses[0].pos_x = 2
        self.warehouses[0].pos_y = 3
        self.warehouses[0].length = 2
        self.warehouses[0].width = 2
        self.warehouses[0].pos_input = [2, 4]
        self.warehouses[0].pos_output = [2, 3]
        self.warehouses[0].input_products = ['empty_w1']
        self.warehouses[0].output_products = ['lager_m1', 'lager_m2', 'lager_m3']
        self.warehouses[0].process_time = 10
        self.warehouses[0].rest_process_time = 10
        self.warehouses[0].factory = self

        self.machines.append(Machine())
        self.machines[0].name = 'M1'
        self.machines[0].pos_x = 5
        self.machines[0].pos_y = 0
        self.machines[0].length = 2
        self.machines[0].width = 2
        self.machines[0].pos_input = [5, 1]
        self.machines[0].pos_output = [5, 0]
        self.machines[0].input_products = ['lager_m1']
        self.machines[0].output_products = ['m1_m3']
        self.machines[0].process_time = 20
        self.machines[0].rest_process_time = 20
        self.machines[0].factory = self

        self.machines.append(Machine())
        self.machines[1].name = 'M2'
        self.machines[1].pos_x = 5
        self.machines[1].pos_y = 3
        self.machines[1].length = 2
        self.machines[1].width = 2
        self.machines[1].pos_input = [5, 4]
        self.machines[1].pos_output = [5, 3]
        self.machines[1].input_products = ['lager_m2', 'm3_m2']
        self.machines[1].output_products = ['empty']
        self.machines[1].process_time = 30
        self.machines[1].rest_process_time = 30
        self.machines[1].factory = self

        self.machines.append(Machine())
        self.machines[2].name = 'M3'
        self.machines[2].pos_x = 6
        self.machines[2].pos_y = 8
        self.machines[2].length = 2
        self.machines[2].width = 2
        self.machines[2].pos_input = [6, 9]
        self.machines[2].pos_output = [6, 8]
        self.machines[2].input_products = ['lager_m3', 'm1_m3']
        self.machines[2].output_products = ['m3_lager']
        self.machines[1].process_time = 15
        self.machines[1].rest_process_time = 15
        self.machines[2].factory = self

        for i in range(6):
            self.loading_stations.append(LoadingStation())
            self.loading_stations[i].name = f'Loading_Station_{i}'
            self.loading_stations[i].pos_x = 0
            self.loading_stations[i].pos_y = i
            self.loading_stations[i].length = 1
            self.loading_stations[i].width = 1
            self.agvs.append(AGV([0, i]))
            self.agvs[i].name = f"{i}"
            self.agvs[i].factory = self

        self.fill_grid()

    def create_temp_factory_machines_6(self):
        self.length = 15
        self.width = 5
        self.no_columns = int(self.length // self.cell_size)
        self.no_rows = int(self.width // self.cell_size)
        self.factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows)).tolist()

        self.product_types['W0_Output'] = dict(length=200, width=200, weight=5)
        self.product_types['M0_Output'] = dict(length=200, width=200, weight=5)
        self.product_types['M1_Output'] = dict(length=200, width=200, weight=5)
        self.product_types['no_Input'] = dict(length=250, width=250, weight=5)
        self.product_types['no_Output'] = dict(length=250, width=250, weight=5)

        self.warehouses.append(Warehouse())
        self.warehouses[0].name = 'W0'
        self.warehouses[0].pos_x = 0
        self.warehouses[0].pos_y = 0
        self.warehouses[0].length = 3
        self.warehouses[0].width = 3
        self.warehouses[0].pos_input = [0, 2]
        self.warehouses[0].pos_output = [2, 2]
        self.warehouses[0].input_products = ['no_Input']
        self.warehouses[0].output_products = ['W0_Output']
        self.warehouses[0].process_time = 10000
        self.warehouses[0].rest_process_time = 10000
        self.warehouses[0].factory = self

        initial_product = self.create_product('W0_Output')
        initial_product.stored_in = self.warehouses[0]
        self.warehouses[0].buffer_output_load.append(initial_product)

        self.warehouses.append(Warehouse())
        self.warehouses[1].name = 'W1'
        self.warehouses[1].pos_x = 12
        self.warehouses[1].pos_y = 0
        self.warehouses[1].length = 3
        self.warehouses[1].width = 3
        self.warehouses[1].pos_input = [12, 2]
        self.warehouses[1].pos_output = [14, 2]
        self.warehouses[1].input_products = ['M1_Output']
        self.warehouses[1].output_products = ['no_Output']
        self.warehouses[1].process_time = 50000000000
        self.warehouses[1].rest_process_time = 50000000000
        self.warehouses[1].factory = self

        self.machines.append(Machine())
        self.machines[0].name = 'M0'
        self.machines[0].pos_x = 4
        self.machines[0].pos_y = 0
        self.machines[0].length = 3
        self.machines[0].width = 3
        self.machines[0].pos_input = [4, 2]
        self.machines[0].pos_output = [6, 2]
        self.machines[0].input_products = ['W0_Output']
        self.machines[0].output_products = ['M0_Output']
        self.machines[0].process_time = 5
        self.machines[0].rest_process_time = 5
        self.machines[0].buffer_input = [1]
        self.machines[0].buffer_output = [6]
        self.machines[0].amount_of_resulting_products = 6
        self.machines[0].factory = self

        self.machines.append(Machine())
        self.machines[1].name = 'M1'
        self.machines[1].pos_x = 8
        self.machines[1].pos_y = 0
        self.machines[1].length = 3
        self.machines[1].width = 3
        self.machines[1].pos_input = [8, 2]
        self.machines[1].pos_output = [10, 2]
        self.machines[1].input_products = ['M0_Output']
        self.machines[1].output_products = ['M1_Output']
        self.machines[1].process_time = 5
        self.machines[1].rest_process_time = 5
        self.machines[1].buffer_input = [1]
        self.machines[1].buffer_output = [1]
        self.machines[1].amount_of_resulting_products = 1
        self.machines[1].factory = self

        for i in range(6):
            self.loading_stations.append(LoadingStation())
            self.loading_stations[i].name = f'Loading_Station_{i}'
            self.loading_stations[i].pos_x = i
            self.loading_stations[i].pos_y = 4
            self.loading_stations[i].length = 1
            self.loading_stations[i].width = 1
            self.agvs.append(AGV([i, 4]))
            self.agvs[i].name = f"{i}"
            self.agvs[i].factory = self

        self.fill_grid()

        for x in range(self.length):
            self.factory_grid_layout[x][3] = Path()


    def fill_grid(self):  # Path integrieren?!
        self.factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows)).tolist()
        for warehouse in self.warehouses:
            self.add_to_grid(warehouse)

        for machine in self.machines:
            self.add_to_grid(machine)

        for loading_station in self.loading_stations:
            self.add_to_grid(loading_station)

    def add_to_grid(self, factor_object):
        for y in range(factor_object.width):
            for x in range(factor_object.length):
                self.factory_grid_layout[factor_object.pos_x + x][factor_object.pos_y + y] = factor_object
                # self.factory_grid_layout[factor_object.pos_y + y][factor_object.pos_x + x] = factor_object

    def delete_from_grid(self, factory_object):
        for y in range(factory_object.width):
            for x in range(factory_object.length):
                self.factory_grid_layout[factory_object.pos_x + x][factory_object.pos_y + y] = 0.0

    def check_collision(self, factory_object):
        for y in range(factory_object.width):
            for x in range(factory_object.length):
                if self.factory_grid_layout[factory_object.pos_x + x][factory_object.pos_y + y] != 0.0 and \
                        factory_object.name != \
                        self.factory_grid_layout[factory_object.pos_x + x][factory_object.pos_y + y].name:
                    # print('COLLISON!!!')
                    return True
        return False

    def check_factory_boundaries(self, factory_object):
        for y in range(factory_object.width):
            for x in range(factory_object.length):
                if factory_object.pos_x + x >= self.length or factory_object.pos_y + y >= self.width:
                    # print('OUT OF BOUNDARIES')
                    return True
        return False

    def check_for_duplicate_names(self, factory_object):
        if type(factory_object) == Machine:
            for i in range(len(self.machines)):
                if factory_object.name == self.machines[i].name and factory_object.id != self.machines[i].id:
                    return True
        elif type(factory_object) == Warehouse:
            for i in range(len(self.warehouses)):
                if factory_object.name == self.warehouses[i].name and factory_object.id != self.warehouses[i].id:
                    return True
        elif type(factory_object) == LoadingStation:
            for i in range(len(self.loading_stations)):
                if (factory_object.name == self.loading_stations[i].name and
                        factory_object.id != self.loading_stations[i].id):
                    return True
        else:
            # print('No duplicate')
            return False

    def get_color_grid(self):
        color_grid = np.ones(shape=(self.no_columns, self.no_rows)).tolist()
        for y in range(self.no_rows):
            for x in range(self.no_columns):
                if self.factory_grid_layout[x][y] == 0.0:
                    color_grid[x][y] = [255, 255, 255]
                else:
                    color_grid[x][y] = self.factory_grid_layout[x][y].get_color()
                    block_type = self.factory_grid_layout[x][y].get_block_type([x, y])
                    if block_type is not None or block_type != "machine_block":
                        if block_type == "input":
                            color_grid[x][y] = [255, 100, 100]
                        elif block_type == "output":
                            color_grid[x][y] = [100, 255, 100]
                        elif block_type == "input_output":
                            color_grid[x][y] = [255, 0, 220]

        return color_grid

    def create_product(self, product_name):
        new_product = Product()
        new_product.id = self.products_id_count
        new_product.length = self.product_types[product_name]['length']
        new_product.width = self.product_types[product_name]['width']
        new_product.weight = self.product_types[product_name]['weight']
        self.products_id_count += 1
        new_product.name = product_name
        new_product.stored_in = self
        self.products.append(new_product)
        return new_product

    def change_product(self, product, product_name):
        product.name = product_name
        product.length = self.product_types[product_name]['length']
        product.width = self.product_types[product_name]['width']
        product.weight = self.product_types[product_name]['weight']


    def get_amount_of_factory_objects(self):
        """
        is used in Class FactoryScene()
        :return: returns the amount of warehouses, machines and loading stations inside the factory
        """
        dim = 0
        for _ in self.warehouses:
            dim += 1
        for _ in self.machines:
            dim += 1
        for _ in self.loading_stations:
            dim += 1
        # print(f'Dimension Distance Matrix = {dim}')
        return dim

    def get_list_of_factory_objects(self):
        """
        is used in Class FactoryScene()
        :return: returns a list with the factory objects in the order: warehouses, machines, loading stations
        """
        list_of_factory_objects = []
        for warehouse in self.warehouses:
            list_of_factory_objects.append(warehouse)
        for machine in self.machines:
            list_of_factory_objects.append(machine)
        for loading_station in self.loading_stations:
            list_of_factory_objects.append(loading_station)
        # print(f'List of Factory_Objects: {list_of_factory_objects}')
        return list_of_factory_objects


    def get_list_of_factory_objects_agvs_warehouse_machines_input_output(self):
        list = []
        for agv in self.agvs:
            if agv.is_free:
                list.append(agv)
        for warehouse in self.warehouses:
            for input_product in warehouse.input_products:
                list.append(warehouse)
            for output_product in warehouse.output_products:
                list.append(warehouse)
        for machine in self.machines:
            for input_product in machine.input_products:
                list.append(machine)
            for output_product in machine.output_products:
                list.append(machine)
        # print(f'List of Factory_Objects: {[object.name for object in list]}')
        return list

    def get_list_of_factory_objects_agvs_warehouse_machines_input_output_product(self):
        pass

    def get_full_list_of_factory_objects_agvs_warehouse_machines_input_output(self):
        list = []
        for agv in self.agvs:
            list.append(agv)
        for warehouse in self.warehouses:
            list.append(warehouse)
            list.append(warehouse)
        for machine in self.machines:
            list.append(machine)
            list.append(machine)
        # print(f'List of Factory_Objects: {[object.name for object in list]}')
        return list

    def get_list_of_factory_objects_loadingstations_warehouses_machines_input_output_agvs(self):
        list = []
        for loading_station in self.loading_stations:
            list.append(loading_station)
        for warehouse in self.warehouses:
            list.append(warehouse)
            list.append(warehouse)
        for machine in self.machines:
            list.append(machine)
            list.append(machine)
        for agv in self.agvs:
            if agv.is_free:
                list.append(agv)
        # print(f'List of Factory_Objects: {list}')
        return list

    def get_list_input_output(self):
        list = []
        for agv in self.agvs:
            if agv.is_free:
                list.append('agv')
        for warehouse in self.warehouses:
            for input_product in warehouse.input_products:
                list.append('input')
            for output_product in warehouse.output_products:
                list.append('output')
        for machine in self.machines:
            for input_product in machine.input_products:
                list.append('input')
            for output_product in machine.output_products:
                list.append('output')
        # print(f'List input/output: {list}')
        return list

    def get_list_input_output_product(self):
        list = []
        i = 0
        for agv in self.agvs:
            if agv.is_free:
                list.append(['agv', None, agv, i])
                i += 1
        for warehouse in self.warehouses:
            for input_product in warehouse.input_products:
                list.append(['input', input_product, warehouse, i])
                i += 1
            for output_product in warehouse.output_products:
                list.append(['output', output_product, warehouse, i])
                i += 1
        for machine in self.machines:
            for input_product in machine.input_products:
                list.append(['input', input_product, machine, i])
                i += 1
            for output_product in machine.output_products:
                list.append(['output', output_product, machine, i])
                i += 1
        # print(f'List input/output/product: {list}')
        return list

    def get_list_input_output_dynamic_vrp_all_objects(self):
        list = []
        for loading_station in self.loading_stations:
            list.append('loading_station')
        for warehouse in self.warehouses:
            list.append('input')
            list.append('output')
        for machine in self.machines:
            list.append('input')
            list.append('output')
        for agv in self.agvs:
            if agv.is_free:
                list.append('agv')
        print(f'List input/output: {list}')
        return list

    def get_num_free_agv_for_vrp(self):
        list_of_free_agvs = []
        for agv in self.agvs:
            if agv.is_free:
                list_of_free_agvs.append(agv)
        return list_of_free_agvs

    def get_amount_of_delivery_nodes(self):
        # returns the amount of warehouses, machines and loading stations inside the factory
        dim = 0
        for warehouse in self.warehouses:
            for input_product in warehouse.input_products:
                dim += 1
            for output_product in warehouse.output_products:
                dim += 1
        for machine in self.machines:
            for input_product in machine.input_products:
                dim += 1
            for output_product in machine.output_products:
                dim += 1
        for agv in self.agvs:
            if agv.is_free:
                dim += 1
        # print(f'Dimension Distance Matrix = {dim}')
        return dim

    def get_agv_needed_for_product(self, product_name, agv):
        """
        Calculates the AGV positioning Matrix for a product delivery
        :param product_name: string
        :param agv: object
        :return: list
        """
        return [int(self.product_types[product_name]['width'] / agv.width + 0.99),
                int(self.product_types[product_name]['length'] / agv.length + 0.99)]

    def dict_to_list(self):
        for key, value in self.product_types.items():
            self.product_types_list.append([key, value['length'], value['width'], value['weight']])
            #print(key)
            #print(value)
        #print(self.product_types_list)
        return self.product_types_list

    def list_to_dict(self, list):
        dict_product_types = {}
        for i in range(len(list)):
            dict_product_types[list[i][0]] = dict(length=list[i][1], width=list[i][2], weight=list[i][3])
        return dict_product_types

    def get_idle_process_times(self):
        dict_times = dict()
        for machine in self.machines:
            dict_times[f'{machine.name}'] = [machine.rest_process_time]

    def shout_down(self):
        for agv in self.agvs:
            agv.thread_running = False

    def get_available_agvs_idx(self):
        agv_list = []
        for i, agv in enumerate(self.agvs):
            if agv.is_free:
                agv_list.append(i)
        return agv_list

    def get_path_queue(self, start, target):
        return self.path_graph.get_predecessor_queue(start, target)

    def print_status(self):
        print("\n________________ FACTORY - STATUS ____________________\n")
        print("----------------AGV - Status----------------")
        for agv in self.agvs:
            print(agv.get_info())

        print("\n--------------Objects - Status -------------")
        for w in self.warehouses:
            print(f"\nName: \t\t\t\t{w.name}")
            print(f"Status: \t\t\t{w.status}")
            print(f"Rest process time: \t{w.rest_process_time}")
            print(f"Output_Buffer: \t\t{[p.name for p in w.buffer_output_load]}")
            print(f"Input_Buffer: \t\t{[p.name for p in w.buffer_input_load]}")
            print(f"Output Products: \t{w.output_products}")
            print(f"Input Products: \t{w.input_products}")

        for m in self.machines:
            print(f"\nName: \t\t\t\t{m.name}")
            print(f"Status: \t\t\t{m.status}")
            print(f"Rest process time: \t{m.rest_process_time}")
            print(f"Output_Buffer: \t\t{[p.name for p in m.buffer_output_load]}")
            print(f"Input_Buffer: \t\t{[p.name for p in m.buffer_input_load]}")
            print(f"Output Products: \t{ m.output_products}")
            print(f"Input Products: \t{m.input_products}")
        # self.path_graph.draw_nx_graph()


        print("__________________________________________________________")



