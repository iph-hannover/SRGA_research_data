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


    def create_default_product_types(self):
        self.product_types['default_product_2'] = dict(length=1000, width=1000, weight=100.0)
        self.product_types['default_product_3'] = dict(length=1500, width=1000, weight=150.0)
        self.product_types['default_product_4'] = dict(length=500, width=1000, weight=50.0)
        #print(self.product_types)
        self.dict_to_list()

    def reset(self):
        for agv in self.agvs:
            agv.reset()
        for machine in self.machines:
            machine.reset()
        for warehouse in self.warehouses:
            warehouse.reset()
        self.products = []

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


    # def get_amount_of_factory_objects(self):
    #     """
    #     is used in Class FactoryScene()
    #     :return: returns the amount of warehouses, machines and loading stations inside the factory
    #     """
    #     dim = 0
    #     for _ in self.warehouses:
    #         dim += 1
    #     for _ in self.machines:
    #         dim += 1
    #     for _ in self.loading_stations:
    #         dim += 1
    #     # print(f'Dimension Distance Matrix = {dim}')
    #     return dim

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



