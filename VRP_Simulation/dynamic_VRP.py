import numpy as np
import pandas as pd
import pickle
import pygame
import csv
import math
import logging
from pulp import *
from itertools import chain, combinations
from copy import deepcopy
import pdb

from FactoryObjects.Factory import Factory
from FactoryObjects.Warehouse import Warehouse
from FactoryObjects.Machine import Machine
from FactoryObjects.Product import Product
from FactoryObjects.LoadingStation import LoadingStation
from FactoryObjects.AGV import AGV
from VRP_Simulation.Factory_Simulation import Factory_Simulation


class VRP_cellAGV:
    """
    This class contains all the information for the VRP.
    """

    def __init__(self, factory: Factory, factory_simulation: Factory_Simulation, use_paths=False, write_data=False, with_prints=True):
        self.factory: Factory = factory
        self.factory_simulation: Factory_Simulation = factory_simulation
        self.project_path = sys.path[1]
        self.agv = AGV()
        self.agv.thread_running = False
        self.agv.length = 500  # length of agv in mm
        self.agv.width = 500  # width of agv in mm
        self.amount_of_nodes = self.factory.get_amount_of_delivery_nodes()
        self.dimension = self.amount_of_nodes
        self.list_of_factory_objects_input_output = (
            self.factory.get_list_of_factory_objects_agvs_warehouse_machines_input_output())
        self.list_agv_input_output = self.factory.get_list_input_output_product()
        # self.distance_matrix = np.zeros(shape=(self.amount_of_nodes, self.amount_of_nodes))
        # self.distance_matrix = self.get_distance_matrix()
        # self.delivery_matrix = np.zeros(shape=(self.amount_of_nodes, self.amount_of_nodes))
        # self.delivery_matrix = self.get_delivery_relationship()
        # self.delivery_matrix_with_agv = np.zeros(shape=(self.amount_of_nodes, self.amount_of_nodes))
        # self.delivery_matrix_with_agv = self.get_amount_of_agv_for_delivery_as_matrix_1_4_6_configuration()
        # Dictionaries für das Speichern der Werte der Binärvariablen in einer Datei
        self.dict_result_1 = dict()
        self.dict_result_2 = dict()
        self.x_values = dict()
        self.x_values_2 = dict()
        self.dict_order_of_transport = dict()
        # Values for Visualisation with pygame
        self.pixel_size = 50
        self.height = self.factory.width
        self.width = self.factory.length
        self.reference_size = self.factory.cell_size * 1000
        self.screen = None

        self.use_paths = use_paths

        self.logger = logging.getLogger()
        if with_prints:
            self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

        self.write_state_output = write_data
        self.write_factory_files = True

    def create_list_of_factory_objects(self):
        """
        This function writes all factory objects (Warehouses, Machines, AGVs) to a list.
        :return: list_of_factory_objects_name
                 A list of all factory objects in the order mentioned above as a list
        """
        list_of_factory_objects_name = []
        j = 0
        for i in self.list_agv_input_output:
            if isinstance(self.list_agv_input_output[j][2], AGV):
                list_of_factory_objects_name.append(self.list_agv_input_output[j][2].name)
            else:
                list_of_factory_objects_name.append(
                    f"{self.list_agv_input_output[j][2].name}_{self.list_agv_input_output[j][0]}")
            j += 1

        # for i in range(self.dimension):
        #     if isinstance(self.list_of_factory_objects_input_output[i], AGV):
        #         list_of_factory_objects_name.append(self.list_of_factory_objects_input_output[i].name)
        #     else:
        #         if self.list_of_factory_objects_input_output[i] != self.list_of_factory_objects_input_output[i - 1]:
        #             list_of_factory_objects_name.append(f'{self.list_of_factory_objects_input_output[i].name}_input')
        #         else:
        #             list_of_factory_objects_name.append(f'{self.list_of_factory_objects_input_output[i].name}_output')

        return list_of_factory_objects_name

    def get_distance_matrix(self):
        """
        Creates a matrix with the distance relationships between individual factory objects I and J.
        :return: distance_matrix as a 2D-np-array
        """
        # Die Matrix distance_matrix enthält die Distanzen aller Maschinen, Warenhäuser und AGVs.
        # Aus Gründen der Vereinfachung wird die Eukild-Distanz verwendet. Mögliche Kollisionen werden ignoriert.
        distance_matrix = np.zeros(shape=(self.amount_of_nodes, self.amount_of_nodes))
        list_of_factory_objects_name = self.create_list_of_factory_objects()
        distance_x = 0
        distance_y = 0

        for i in range(self.dimension):
            for j in range(self.dimension):
                # 1st if: AGV - AGV
                if ((type(self.list_of_factory_objects_input_output[i]) == AGV and
                     self.list_of_factory_objects_input_output[i].is_free) and
                        (type(self.list_of_factory_objects_input_output[j]) == AGV and
                         self.list_of_factory_objects_input_output[j].is_free)):
                    distance_x = abs(self.list_of_factory_objects_input_output[i].pos_x
                                     - self.list_of_factory_objects_input_output[j].pos_x)
                    distance_y = abs(self.list_of_factory_objects_input_output[i].pos_y
                                     - self.list_of_factory_objects_input_output[j].pos_y)
                    # self.logger.debug(f'{self.list_of_factory_objects_input_output[i].name}: '
                    #       f'pos_x = {self.list_of_factory_objects_input_output[i].pos_x}, '
                    #       f'pos_y = {self.list_of_factory_objects_input_output[i].pos_y}')
                # 2nd if: AGV - Machine/Warehouse
                elif ((type(self.list_of_factory_objects_input_output[i]) == AGV and
                       self.list_of_factory_objects_input_output[i].is_free) and
                      (type(self.list_of_factory_objects_input_output[j]) == Warehouse or
                       type(self.list_of_factory_objects_input_output[j]) == Machine)):
                    if self.list_agv_input_output[j][0] == 'input':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_x -
                                         self.list_of_factory_objects_input_output[j].pos_input[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_y -
                                         self.list_of_factory_objects_input_output[j].pos_input[1])
                    elif self.list_agv_input_output[j][0] == 'output':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_x -
                                         self.list_of_factory_objects_input_output[j].pos_output[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_y -
                                         self.list_of_factory_objects_input_output[j].pos_output[1])
                # 3rd if: Machine/Warehouse - AGV
                elif ((type(self.list_of_factory_objects_input_output[i]) == Warehouse or
                       type(self.list_of_factory_objects_input_output[i]) == Machine) and
                      (type(self.list_of_factory_objects_input_output[j]) == AGV and self.list_of_factory_objects_input_output[j].is_free)):
                    if self.list_agv_input_output[i][0] == 'input':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_input[0] -
                                         self.list_of_factory_objects_input_output[j].pos_x)
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_input[1] -
                                         self.list_of_factory_objects_input_output[j].pos_y)
                    elif self.list_agv_input_output[i][0] == 'output':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_output[0] -
                                         self.list_of_factory_objects_input_output[j].pos_x)
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_output[1] -
                                         self.list_of_factory_objects_input_output[j].pos_y)
                # 4th if: Machine/Warehouse - Machine/Warehouse
                elif ((type(self.list_of_factory_objects_input_output[i]) == Warehouse or
                       type(self.list_of_factory_objects_input_output[i]) == Machine) and
                      (type(self.list_of_factory_objects_input_output[i]) == Warehouse or
                       type(self.list_of_factory_objects_input_output[i]) == Machine)):
                    if self.list_agv_input_output[i][0] == 'input' and self.list_agv_input_output[j][0] == 'output':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_input[0] -
                                         self.list_of_factory_objects_input_output[j].pos_output[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_input[1] -
                                         self.list_of_factory_objects_input_output[j].pos_output[1])
                    elif self.list_agv_input_output[i][0] == 'input' and self.list_agv_input_output[j][0] == 'input':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_input[0] -
                                         self.list_of_factory_objects_input_output[j].pos_input[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_input[1] -
                                         self.list_of_factory_objects_input_output[j].pos_input[1])
                    elif self.list_agv_input_output[i][0] == 'output' and self.list_agv_input_output[j][0] == 'input':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_output[0] -
                                         self.list_of_factory_objects_input_output[j].pos_input[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_output[1] -
                                         self.list_of_factory_objects_input_output[j].pos_input[1])
                    elif self.list_agv_input_output[i][0] == 'output' and self.list_agv_input_output[j][0] == 'output':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_output[0] -
                                         self.list_of_factory_objects_input_output[j].pos_output[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_output[1] -
                                         self.list_of_factory_objects_input_output[j].pos_output[1])
                # Manhattan-Distance
                # distance_matrix[i][j] = distance_x + distance_y
                # Euklid-Distance
                distance_matrix[i][j] = math.sqrt((distance_x * distance_x) + (distance_y * distance_y)) + 0.01
        distance_matrix_pd = pd.DataFrame(distance_matrix, columns=list_of_factory_objects_name,
                                          index=list_of_factory_objects_name)
        # distance_matrix_pd.to_csv(self.project_path + '/data/Current_Factory/VRP/distance_matrix.csv', sep=';')
        return distance_matrix.tolist()

    def get_weighted_distance_matrix(self):
        # Entfernungsmatrix reinladen und diese mit einer Gewichtung multiplizieren
        # Auftrag weitere Eigenschaften geben
        distance_matrix = np.zeros(shape=(self.amount_of_nodes, self.amount_of_nodes))
        distance_x = 0
        distance_y = 0

        for i in range(self.dimension):
            for j in range(self.dimension):
                # 1st if: AGV - AGV
                if (type(self.list_of_factory_objects_input_output[i]) == AGV and
                        type(self.list_of_factory_objects_input_output[j]) == AGV):
                    distance_x = abs(self.list_of_factory_objects_input_output[i].pos_x
                                     - self.list_of_factory_objects_input_output[j].pos_x)
                    distance_y = abs(self.list_of_factory_objects_input_output[i].pos_y
                                     - self.list_of_factory_objects_input_output[j].pos_y)
                # 2nd if: AGV - Machine/Warehouse
                elif (type(self.list_of_factory_objects_input_output[i]) == AGV and
                      (type(self.list_of_factory_objects_input_output[j]) == Warehouse or
                       type(self.list_of_factory_objects_input_output[j]) == Machine)):
                    if self.list_agv_input_output[j][0] == 'input':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_x -
                                         self.list_of_factory_objects_input_output[j].pos_input[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_y -
                                         self.list_of_factory_objects_input_output[j].pos_input[1])
                    elif self.list_agv_input_output[j][0] == 'output':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_x -
                                         self.list_of_factory_objects_input_output[j].pos_output[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_y -
                                         self.list_of_factory_objects_input_output[j].pos_output[1])
                # 3rd if: Machine/Warehouse - AGV
                elif ((type(self.list_of_factory_objects_input_output[i]) == Warehouse or
                       type(self.list_of_factory_objects_input_output[i]) == Machine) and
                      type(self.list_of_factory_objects_input_output[j]) == AGV):
                    if self.list_agv_input_output[i][0] == 'input':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_input[0] -
                                         self.list_of_factory_objects_input_output[j].pos_x)
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_input[1] -
                                         self.list_of_factory_objects_input_output[j].pos_y)
                    elif self.list_agv_input_output[i][0] == 'output':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_output[0] -
                                         self.list_of_factory_objects_input_output[j].pos_x)
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_output[1] -
                                         self.list_of_factory_objects_input_output[j].pos_y)
                # 4th if: Machine/Warehouse - Machine/Warehouse
                elif ((type(self.list_of_factory_objects_input_output[i]) == Warehouse or
                       type(self.list_of_factory_objects_input_output[i]) == Machine) and
                      (type(self.list_of_factory_objects_input_output[i]) == Warehouse or
                       type(self.list_of_factory_objects_input_output[i]) == Machine)):
                    if self.list_agv_input_output[i][0] == 'input' and self.list_agv_input_output[j][0] == 'output':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_input[0] -
                                         self.list_of_factory_objects_input_output[j].pos_output[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_input[1] -
                                         self.list_of_factory_objects_input_output[j].pos_output[1])
                    elif self.list_agv_input_output[i][0] == 'input' and self.list_agv_input_output[j][0] == 'input':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_input[0] -
                                         self.list_of_factory_objects_input_output[j].pos_input[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_input[1] -
                                         self.list_of_factory_objects_input_output[j].pos_input[1])
                    elif self.list_agv_input_output[i][0] == 'output' and self.list_agv_input_output[j][0] == 'input':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_output[0] -
                                         self.list_of_factory_objects_input_output[j].pos_input[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_output[1] -
                                         self.list_of_factory_objects_input_output[j].pos_input[1])
                    elif self.list_agv_input_output[i][0] == 'output' and self.list_agv_input_output[j][0] == 'output':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_output[0] -
                                         self.list_of_factory_objects_input_output[j].pos_output[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_output[1] -
                                         self.list_of_factory_objects_input_output[j].pos_output[1])
                # Manhattan-Distance
                # distance_matrix[i][j] = distance_x + distance_y
                # Euklid-Distance
                distance_matrix[i][j] = math.sqrt((distance_x * distance_x) + (distance_y * distance_y)) + 0.01

        for i in range(self.dimension):
            for j in range(self.dimension):
                if (type(self.list_of_factory_objects_input_output[i]) == Machine or
                        type(self.list_of_factory_objects_input_output[i]) == Warehouse):
                    if self.list_agv_input_output[i][0] == 'output':
                        if (type(self.list_of_factory_objects_input_output[j]) == Machine or
                                type(self.list_of_factory_objects_input_output[j]) == Warehouse):
                            if self.list_agv_input_output[j][0] == 'input':
                                if self.list_agv_input_output[i][1] == self.list_agv_input_output[j][1]:  # productcheck
                                    if self.list_of_factory_objects_input_output[i].has_product(self.list_agv_input_output[i][1]):
                                        prio = self.get_delivery_priorities(
                                                    self.list_of_factory_objects_input_output[i],
                                                    self.list_of_factory_objects_input_output[j])
                                        self.logger.debug(f'{i} - {j} mit Priorität = {prio}')
                                        distance_matrix[i][j] = prio * distance_matrix[i][j]

        return distance_matrix.tolist()

    def get_delivery_priorities(self, start_object, end_object):
        """
        This function adds priorities to the deliveries.
        Therefore, the function "get_buffer_status" from machines will be used.
        Routes with high priorities must be prioritised.
        :return:
        """
        if type(start_object) == Machine:
            output_priority_start = start_object.get_buffer_status()[1]
        else:
            output_priority_start = 1
        if type(end_object) == Machine:
            input_priority_end = end_object.get_buffer_status()[0]
        else:
            input_priority_end = 1
        # self.logger.debug(f'Start_Priority: {output_priority_start}, End_Priority: {input_priority_end}')
        if input_priority_end == 0 or output_priority_start == 0:
            return 10000000000
        else:
            return 1 / (output_priority_start * input_priority_end)

    def get_delivery_relationship(self):
        '''
        Creates a matrix with the delivery relationships between individual factory objects I and J.
        If a delivery from I to J takes place, a '1' is set at the corresponding position in the matrix.
        The matrix is saved as a CSV file under the path: data/Current_Factory/VRP/delivery_matrix.csv
        :return: delivery_matrix as a 2D-np-array
        '''
        list_agv_input_output = self.factory.get_list_input_output_product()
        delivery_matrix = np.zeros(shape=(self.amount_of_nodes, self.amount_of_nodes))
        list_of_factory_objects_name = self.create_list_of_factory_objects()
        m = 1
        for i in range(self.dimension):
            for j in range(self.dimension):
                if (type(self.list_of_factory_objects_input_output[i]) == Machine or
                        type(self.list_of_factory_objects_input_output[i]) == Warehouse):
                    if self.list_agv_input_output[i][0] == 'output':
                        if (type(self.list_of_factory_objects_input_output[j]) == Machine or
                                type(self.list_of_factory_objects_input_output[j]) == Warehouse):
                            if self.list_agv_input_output[j][0] == 'input':
                                if self.list_agv_input_output[i][1] == self.list_agv_input_output[j][1]:  # productcheck
                                    if self.list_of_factory_objects_input_output[i].has_product(
                                            self.list_agv_input_output[i][1]):
                                        self.logger.debug(
                                            '---------------------------------------------------------------------------')
                                        self.logger.debug(f'{m:03d}. Lieferbeziehung gefunden! \n'
                                                          f'Lieferung von {self.list_of_factory_objects_input_output[i].name} '
                                                          f'(Knoten {i}) nach {self.list_of_factory_objects_input_output[j].name} '
                                                          f'(Knoten {j}) \n'
                                                          f'Geliefertes Produkt: {self.list_agv_input_output[i][1]}')
                                        if (type(self.list_of_factory_objects_input_output[j]) == Machine and
                                                self.list_of_factory_objects_input_output[j].get_buffer_status()[0] != 0
                                                or self.list_of_factory_objects_input_output[j].status == 'idle'):
                                            # Kann nur beliefert werden, wenn der Buffer von Machine[j] frei ist
                                            delivery_matrix[i][j] = 1
                                        elif type(self.list_of_factory_objects_input_output[j]) == Warehouse:
                                            delivery_matrix[i][j] = 1
                                        if (self.check_amount_of_agvs_for_transport_1_4_6_configuration(
                                                self.factory.product_types[self.list_agv_input_output[i][1]]) >
                                                len(self.factory.get_num_free_agv_for_vrp())):
                                            delivery_matrix[i][j] = 0
                                            # self.delivery_matrix_with_agv[i][j] = delivery_matrix_with_agv[i][j]
                                        # else:
                                        #     delivery_matrix_with_agv[i][
                                        #         j] = self.check_amount_of_agvs_for_transport_1_4_6_configuration(
                                        #         self.factory.product_types[output_product])
                                        m += 1

        delivery_matrix_pd = pd.DataFrame(delivery_matrix, columns=list_of_factory_objects_name,
                                                   index=list_of_factory_objects_name)
        if self.write_factory_files:
            delivery_matrix_pd.to_csv(
                self.project_path + '/data/Current_Factory/VRP/delivery_matrix.csv', sep=';')
        return delivery_matrix.tolist()

    def get_amount_of_agv_for_delivery_as_matrix_free_configuration(self):
        """
        Creates a matrix with the delivery relationships between individual factory objects I and J,
        taking into account the number of AGVs required for transportation.
        If a delivery from I to J takes place, the number of AGVs required is set at the corresponding position in the
        matrix.
        The matrix is saved as a CSV file under the path: data/Current_Factory/VRP/delivery_matrix_with_agv.csv
        :return: delivery_matrix_with_agv as a 2D-np-array
        """
        delivery_matrix_with_agv = np.zeros(shape=(self.amount_of_nodes, self.amount_of_nodes))
        list_of_factory_objects_name = self.create_list_of_factory_objects()
        m = 1
        for i in range(self.dimension):
            for j in range(self.dimension):
                if (type(self.list_of_factory_objects_input_output[i]) == Machine or
                        type(self.list_of_factory_objects_input_output[i]) == Warehouse):
                    if self.list_agv_input_output[i][0] == 'output':
                        if (type(self.list_of_factory_objects_input_output[j]) == Machine or
                                type(self.list_of_factory_objects_input_output[j]) == Warehouse):
                            if self.list_agv_input_output[j][0] == 'input':
                                if self.list_agv_input_output[i][1] == self.list_agv_input_output[j][1]:  # productcheck
                                    if self.list_of_factory_objects_input_output[i].has_product(self.list_agv_input_output[i][1]):
                                        self.logger.debug(
                                            '---------------------------------------------------------------------------')
                                        self.logger.debug(f'{m:03d}. Lieferbeziehung gefunden! \n'
                                              f'Lieferung von {self.list_of_factory_objects_input_output[i].name} '
                                              f'(Knoten {i}) nach {self.list_of_factory_objects_input_output[j].name} '
                                              f'(Knoten {j}) \n'
                                              f'Geliefertes Produkt: {self.list_agv_input_output[i][1]}')
                                        if (type(self.list_of_factory_objects_input_output[j]) == Machine and
                                                self.list_of_factory_objects_input_output[
                                                    j].get_buffer_status()[0] != 0 or
                                                self.list_of_factory_objects_input_output[j].status == 'idle'):
                                            # Kann nur beliefert werden, wenn der Buffer von Machine[j] frei ist
                                            delivery_matrix_with_agv[i][
                                                j] = self.check_amount_of_agvs_for_transport_free_configuration(
                                                self.factory.product_types[self.list_agv_input_output[i][1]])
                                        elif type(self.list_of_factory_objects_input_output[j]) == Warehouse:
                                            delivery_matrix_with_agv[i][
                                                j] = self.check_amount_of_agvs_for_transport_free_configuration(
                                                self.factory.product_types[self.list_agv_input_output[i][1]])
                                        if self.check_amount_of_agvs_for_transport_free_configuration(
                                                self.factory.product_types[self.list_agv_input_output[i][1]]) > len(
                                            self.factory.get_num_free_agv_for_vrp()):
                                            delivery_matrix_with_agv[i][j] = 0
                                        m += 1
        delivery_matrix_with_agv_pd = pd.DataFrame(delivery_matrix_with_agv, columns=list_of_factory_objects_name,
                                                   index=list_of_factory_objects_name)
        # delivery_matrix_with_agv_pd.to_csv(
        #     self.project_path + '/data/Current_Factory/VRP/delivery_matrix_with_agv.csv', sep=';')
        return delivery_matrix_with_agv.tolist()

    def check_amount_of_agvs_for_transport_free_configuration(self, product):
        """
        Calculates the amount of agv, which is necessary to transport a product from I to J.
        Depending on the length of the product, it is determined how many AGVs have to travel one behind the other.
        Depending on the width of the product, the number of AGVs that must travel side by side is determined.
        :param product: product which is transported from I to J
        :return: amount_of_agv as an int
        """
        length_of_product = product['length']
        width_of_product = product['width']
        length_of_agv = self.agv.length  # / 1000
        width_of_agv = self.agv.width  # / 1000
        length_ratio = length_of_product / length_of_agv
        width_ratio = width_of_product / width_of_agv
        agv_in_a_row = math.ceil(length_ratio)
        agv_side_by_side = math.ceil(width_ratio)
        amount_of_agv = agv_in_a_row * agv_side_by_side
        return amount_of_agv

    def get_amount_of_agv_for_delivery_as_matrix_1_4_6_configuration(self):
        """
        Creates a matrix with the delivery relationships between individual factory objects I and J,
        taking into account the number of AGVs required for transportation.
        If a delivery from I to J takes place, the number of AGVs required is set at the corresponding position in the
        matrix.
        The matrix is saved as a CSV file under the path: data/Current_Factory/VRP/delivery_matrix_with_agv.csv
        :return: delivery_matrix_with_agv as a 2D-np-array
        """
        delivery_matrix_with_agv = np.zeros(shape=(self.amount_of_nodes, self.amount_of_nodes))
        list_of_factory_objects_name = self.create_list_of_factory_objects()
        m = 1
        for i in range(self.dimension):
            for j in range(self.dimension):
                if (type(self.list_of_factory_objects_input_output[i]) == Machine or
                        type(self.list_of_factory_objects_input_output[i]) == Warehouse):
                    if self.list_agv_input_output[i][0] == 'output':
                        if (type(self.list_of_factory_objects_input_output[j]) == Machine or
                                type(self.list_of_factory_objects_input_output[j]) == Warehouse):
                            if self.list_agv_input_output[j][0] == 'input':
                                if self.list_agv_input_output[i][1] == self.list_agv_input_output[j][1]:  # productcheck
                                    if self.list_of_factory_objects_input_output[i].has_product(
                                            self.list_agv_input_output[i][1]):
                                        self.logger.debug(
                                            '---------------------------------------------------------------------------')
                                        self.logger.debug(f'{m:03d}. Lieferbeziehung gefunden! \n'
                                                          f'Lieferung von {self.list_of_factory_objects_input_output[i].name} '
                                                          f'(Knoten {i}) nach {self.list_of_factory_objects_input_output[j].name} '
                                                          f'(Knoten {j}) \n'
                                                          f'Geliefertes Produkt: {self.list_agv_input_output[i][1]}')
                                        if (type(self.list_of_factory_objects_input_output[j]) == Machine and
                                                self.list_of_factory_objects_input_output[j].get_buffer_status()[0] != 0
                                                or self.list_of_factory_objects_input_output[j].status == 'idle'):
                                            # Kann nur beliefert werden, wenn der Buffer von Machine[j] frei ist
                                            delivery_matrix_with_agv[i][
                                                j] = self.check_amount_of_agvs_for_transport_1_4_6_configuration(
                                                self.factory.product_types[self.list_agv_input_output[i][1]])
                                        elif type(self.list_of_factory_objects_input_output[j]) == Warehouse:
                                            delivery_matrix_with_agv[i][
                                                j] = self.check_amount_of_agvs_for_transport_1_4_6_configuration(
                                                self.factory.product_types[self.list_agv_input_output[i][1]])
                                        if (self.check_amount_of_agvs_for_transport_1_4_6_configuration(
                                                self.factory.product_types[self.list_agv_input_output[i][1]]) >
                                                len(self.factory.get_num_free_agv_for_vrp())):
                                            delivery_matrix_with_agv[i][j] = 0
                                            # self.delivery_matrix_with_agv[i][j] = delivery_matrix_with_agv[i][j]
                                        # else:
                                        #     delivery_matrix_with_agv[i][
                                        #         j] = self.check_amount_of_agvs_for_transport_1_4_6_configuration(
                                        #         self.factory.product_types[output_product])
                                        m += 1

        delivery_matrix_with_agv_pd = pd.DataFrame(delivery_matrix_with_agv, columns=list_of_factory_objects_name,
                                                   index=list_of_factory_objects_name)
        if self.write_factory_files:
            delivery_matrix_with_agv_pd.to_csv(
                self.project_path + '/data/Current_Factory/VRP/delivery_matrix_with_agv.csv', sep=';')
        return delivery_matrix_with_agv.tolist()

    def check_amount_of_agvs_for_transport_1_4_6_configuration(self, product):
        """
        Calculates the amount of agv, which is necessary to transport a product from I to J.
        Depending on the length of the product, it is determined how many AGVs have to travel one behind the other.
        Depending on the width of the product, the number of AGVs that must travel side by side is determined.
        :param product: product which is transported from I to J
        :return: amount_of_agv as an int
        """
        length_of_product = product['length']
        width_of_product = product['width']
        length_of_agv = self.agv.length  # / 1000
        width_of_agv = self.agv.width  # / 1000
        length_ratio = length_of_product / length_of_agv
        width_ratio = width_of_product / width_of_agv
        if length_ratio <= 1:
            if width_ratio <= 1:
                agv_in_a_row = 1
                agv_side_by_side = 1
                amount_of_agv = agv_in_a_row * agv_side_by_side
                return amount_of_agv
        elif length_ratio <= 2:
            if width_ratio <= 2:
                agv_in_a_row = 2
                agv_side_by_side = 2
                amount_of_agv = agv_in_a_row * agv_side_by_side
                return amount_of_agv
        elif length_ratio <= 3:
            if width_ratio <= 2:
                agv_in_a_row = 3
                agv_side_by_side = 2
                amount_of_agv = agv_in_a_row * agv_side_by_side
                return amount_of_agv
        else:
            return None

        agv_in_a_row = math.ceil(length_ratio)
        agv_side_by_side = math.ceil(width_ratio)
        amount_of_agv = agv_in_a_row * agv_side_by_side
        return amount_of_agv

    def cVRPPDmDnR(self, num_AGVs, D, Q, T):
        ###################################
        # Eingabegrößen des Modells
        ###################################

        startzeit = time()

        K = []  # Anzahl der Depots, wo die einzelnen FTF stehen - jedes FTF hat ein Depot
        I = []  # Orte/Knoten, die besucht werden (inkl. Depots, Abhol- und Lieferpunkten), wird aus der Distanzmatrix berechnet
        J = []  # Orte/Knoten der Jobs (inkl. Abhol- und Lieferpunkten)
        A = num_AGVs  # Anzahl der zur Verfügung stehenden Fahrzeuge
        v = 1  # Geschwindigkeit der FTF in m/s
        M = 1000000  # Große Zahl für oder Vergleich Startbedingung mit Q_counter
        Q_counter = 0  # Zähler, der alle Routen z#hlt, für die mehr als 1 Fahrzeug benötigt wird.
        QQ = []  # Abholknoten
        w = 1

        self.x_values = dict()
        self.x_values_2 = dict()

        for i in range(A):
            K.append(i)
        for i in range(len(D)):
            I.append(i)
        num_locations = len(D)
        J = list(set(I).difference(K))

        # for j in J:
        #     for l in J:
        #         if Q[j][l] > 0:
        #             QQ.append(j)

        self.logger.debug(f"K = {K}")
        self.logger.debug(f"I = {I}")
        self.logger.debug(f"J = {J}")
        self.logger.debug(f"D = {D}")
        self.logger.debug(f"Q = {Q}")
        # self.logger.debug(f"QQ = {QQ}")
        self.logger.debug(f"T = {T}")


        ##################################
        ###  Formulierung des Modells  ###
        ##################################

        ###################
        # Initialisierung #
        ###################

        # cellular VRP with Pickups and Delivery and a single Depot
        model = LpProblem("cVRPPDmDnR", LpMinimize)

        # Modellvariablen
        x = LpVariable.dicts("x", [(i, j, k) for i in I for j in I for k in K], 0, 1, LpBinary)
        # x_(i, j, k) ist eine Binärvariable, die 1 ist, wenn das Transportfahrzeug k von Ort i direkt zu Ort j fährt; sonst 0.
        u = LpVariable.dicts("u", [i for i in J], 1, len(J), LpInteger)
        # u_(i) gibt die Reihenfolge an, in welcher die Knoten i besucht werden

        # original
        arrival_time = LpVariable.dicts("arrival_time", [(i, j, k) for i in I for j in I for k in I], 0, None,
                                        LpContinuous)
        # # arrival_time_(j, k) gibt die Zeit an, zu welcher das Fahrzeug k spätestens am Knoten j angekommen ist
        max_arrival_time = LpVariable.dicts("max_arrival_time", [(i, j) for i in I for j in I], 0, None, LpContinuous)
        # # Die max_arrival_time_(j) ist die Zeit des Fahrzeugs, welches zuletzt am Knoten j ankommt
        wait_time = LpVariable.dicts("wait_time", [(i, j, k) for i in I for j in I for k in K], 0, None, LpContinuous)
        # # Die wait_time_(j, k) ist die Zeit, die ein Fahrzeug k maximal auf andere Fahrzeuge am Knoten k wartet.
        max_total_arrival_time = LpVariable("transport_duration", 0, None, LpContinuous)
        # Die max_total_arrival_time ist die Dauer, die der gesamte Transport in Anspruch nimmt.
        #
        # idle_time = LpVariable.dicts("idle_time", [(i, j, k) for i in I for j in I for k in I], 0, None,
        #                                 LpContinuous)
        # # # idle_time_(j, k) gibt die Zeit an, die eine Maschine bei Ankunft eines Fahrzeugs im idle-Zustand ist
        # max_idle_time = LpVariable.dicts("max_idle_time", [(i, j) for i in I for j in I], 0, None, LpContinuous)
        # # # Die max_idle_time_(j) ist die Zeit des Fahrzeugs, welches zuletzt am Knoten j ankommt


        ################
        # Zielfunktion #
        ################

        model += lpSum(D[i][j] * x[(i, j, k)] for i in I for j in I for k in K), "Minimize_Total_Distance"

        # model += max_total_arrival_time, "Minimize_Arrival_Time"

        # model += w * (lpSum(D[i][j] * x[(i, j, k)] for i in I for j in I for k in K)) + (1 - w) * max_total_arrival_time

        # model += lpSum(max_idle_time[(i, j)] for i in J for j in J), 'Minimize_Idle_Times'

        ####################
        # Nebenbedingungen #
        ####################

        ################################################################################################################
        # Depots
        # Fahrzeug k startet im Depot h == k und kann nur im Depot bleiben oder zu einem Abholknoten fahren.
        for k in range(A):
            model += (lpSum(x[(k, j, k)] for j in I) == 1, f'Start_im_Depot_{k}_Fahrzeug_{k}')
            # model += ((lpSum(x[(k, k, k)]) + lpSum(x[(k, j, k)] for j in QQ)) == 1,
            #           f'Entweder_bleibt_{k}_im Depot_oder_faehrt_zu_{QQ}')

        # Von jedem Depot kann am Anfang maximal ein Fahrzeug starten.
        for i in K:
            model += (lpSum(x[(i, j, k)] for j in I for k in K) <= 1, f'Anfangszustand_Depot_{i}')

        ################################################################################################################
        # Reihenfolge des Transports zur Verhinderung von Deadlocks
        for k in K:
            for i in J:
                for j in J:
                    if i != j:
                        model += u[i] - u[j] + (len(J) * x[(i, j, k)]) <= len(
                            J) - 1, f"MTZ_Subtour_{i}_{j}_Fahrzeug_{k}"

        ################################################################################################################
        # Flusskonservierung - Fluss von Fahrzeugen in den Knoten muss größer oder gleich dem Fluss von Fahrzeugen
        #                      aus dem Knoten sein - außer für das Depot. Dadurch müssen Fahrzeuge nicht ins Depot zurück.
        for j in J:
            for k in range(A):
                model += (lpSum(x[(i, j, k)] for i in I) - lpSum(x[(j, h, k)] for h in I) >= 0,
                          f"Flusskonservierung_Knoten_{j}_Fahrzeug_{k}")

        ################################################################################################################
        # Subtour-Eliminierungsbedingungen (SEC)
        # def subtour_elimination_constraints(model, x, J, K):
        #     """Fügt Subtour-Eliminierungsbedingungen zur DFJ-Formulierung hinzu für jedes Fahrzeug."""
        #     for k in K:
        #         for size in range(2, len(J)):  # Alle Teilmengen von 2 bis n-1 Knoten
        #             for S in combinations(J, size):
        #                 # print(f'k = {k} - S = {S}')
        #                 model += pulp.lpSum(x[(i, j, k)] for i in S for j in S if i != j) <= len(
        #                     S) - 1, f"Subtour_Elimination_S_{S}_Fahrzeug_{k}"
        #
        # # Subtour-Eliminierung hinzufügen
        # subtour_elimination_constraints(model, x, J, K)

        ####################################################################################################################
        # Jeder Lieferung müssen die richtige Anzahl an AGV zugeordnet werden.
        for i in J:
            for j in J:
                if i != j:
                    if Q[i][j] > 0:
                        model += (lpSum(x[(i, j, k)] for k in K) >= Q[i][j],
                                  f'Anzahl_AGV_pro_Lieferung_Knoten_{i}_nach_Knoten_{j}')

        ####################################################################
        # Die Anzahl an AGVs, die die Depots verlassen und zu den Abholorten fahren, muss passen.
        # Diese Nebenbedingung gilt für mehrere Abholorte und mehrere Depots.
        model += (lpSum(x[(i, j, k)] for i in K for j in J for k in K) >= max(Q[i][j] for i in I for j in I),
                  f'Verlassen_des_Depots')

        # ######################################
        # # Anzahl der Fahrzeuge berücksichtigen.
        # # Die Anzahl der Fahrzeuge, die vom Ort i zu Ort j fahren, darf die max. Anzahl an Fahrzeugen nicht überschreiten.
        # for i in I:
        #     for j in I:
        #         model += (
        #             lpSum(x[(i, j, k)] for k in range(A)) <= A, f'Max_Anzahl_Fahrzeuge_von_Knoten_{i}_nach_Knoten_{j}')
        #
        # model.writeLP('model')  # Ausgabe des Modells als Datei

        ################################################################################################################
        # # Berücksichtigung von Zeiten
        #
        # # Ankunftszeit am Depot ist immer 0
        # for k in K:
        #     model += arrival_time[(k, k, k)] == 0, f"Ankunftszeit_Im_Depot_{k}_ist_0.0"
        #
        # # Ankunftszeit am ersten Knoten nach Verlassen des Depots
        # for k in K:
        #     for j in J:
        #         # Ankunftszeit-Bedingung nur, wenn das Fahrzeug tatsächlich von i nach j fährt
        #         model += (arrival_time[(k, j, k)] == D[k][j] * x[(k, j, k)] / v,
        #                   (f"Ankunftszeit_Fahrzeug_{k}_von_Depot_{k}_zu_erstem_Knoten_{j}"))
        #
        # # Die Ankunftszeit eines Fahrzeugs an einem Knoten hängt von der Ankunftszeit am vorherigen Knoten, der Fahrzeit
        # # zwischen den Knoten und der Wartezeit am vorherigen Knoten ab.
        # for i in I:
        #     for j in I:
        #         for h in I:
        #             if i != j and h != i and h != j:
        #                 for k in K:
        #                     if i != k:
        #                         # Ankunftszeit-Bedingung nur, wenn das Fahrzeug tatsächlich von i nach j fährt
        #                         model += arrival_time[(i, j, k)] >= arrival_time[(h, i, k)] + wait_time[(i, j, k)] + \
        #                                  D[i][j] * x[
        #                                      (i, j, k)] / v - M * (1 - x[(i, j, k)]), (
        #                             f"Ankunftszeit_Fahrzeug_{k}_von_Knoten_"
        #                             f"{i}_zu_Knoten_{j}_ausgehend_von_Knoten_{h}")
        #                         # if T[i][j] > 0:
        #                         #     model += idle_time[(i, j, k)] == arrival_time[(i, j, k)] + T[i][j], f'Idle_Time_Fahrzeug_{k}_von_Knoten_{i}_zu_Knoten_{j}_ausgehend_von_Knoten_{h}'
        #
        # # Die Ankunftszeit des Fahrzeugs, welches am längsten benötigt, ist entscheidend für die Transportdauer
        # for i in J:
        #     for j in J:
        #         for k in K:
        #             # Maximale Ankunftszeit nur für Fahrzeuge, die tatsächlich zu Knoten j fahren
        #             model += max_arrival_time[(i, j)] >= arrival_time[
        #                 (i, j, k)], f"Max_Ankunftszeit_Fahrzeug_{k}_Knoten_{j}_ausgehend_von_Knoten_{i}"
        #             # model += max_idle_time[(i, j)] >= idle_time[(i, j, k)], f'Max_IDLE_Time_Fahrzeug_{k}_Knoten_{j}_ausgehend_von_Knoten_{i}'
        #             model += max_total_arrival_time >= arrival_time[
        #                 (i, j, k)], f"Max_Ankunftszeit_Beschraenkung_{i}_{j}_{k}"
        #
        # # Die Wartezeit eines Fahrzeugs ist die maximale Ankunftszeit minus der eigenen Ankunftszeit
        # # for j in J:
        # #     for k in K:
        # #         # Wartezeit wird nur berechnet, wenn das Fahrzeug tatsächlich zu Knoten j fährt
        # #         model += (wait_time[(j, k)] >= max_arrival_time[j] - arrival_time[(j, k)] -  M * (
        # #                 1 - lpSum(x[(i, j, k)] for i in I)), f"Wartezeit_Fahrzeug_{k}_Knoten_{j}")
        #
        # for j in I:
        #     for i in I:
        #         for h in I:
        #             if i != j and h != i and h != j:
        #                 for k in K:
        #                     # Wartezeit wird nur berechnet, wenn das Fahrzeug tatsächlich zu Knoten j fährt
        #                     model += (
        #                     wait_time[(i, j, k)] >= max_arrival_time[(i, j)] - arrival_time[(h, i, k)] - D[i][j] / v
        #                     - M * (1 - x[(h, i, k)]),
        #                     f"Wartezeit_Fahrzeug_{k}_von_Knoten_{i}_nach_Knoten_{j}_ausgehend_von_Knoten_{h}")

        # Problem lösen
        print("SOLVING PROBLEM...")
        solver = CPLEX_CMD()
        model.to_json('model.json')
        # model.solve(PULP_CBC_CMD(msg=True, gapRel=1))
        model.solve(GLPK_CMD(msg=True, timeLimit=600, mip=True))

        endzeit = time()
        dauer = endzeit - startzeit
        # print(f"Dauer Lösung VRP: {dauer}")

        result = {
            'status': LpStatus[model.status],
            'objective': value(model.objective),
            'variables': {v.name: v.varValue for v in model.variables()}
        }

        for v in model.variables():
            if v.varValue is not None:
                if v.varValue > 0:
                    print(v.name, "=", v.varValue)
        for v in model.variables():
            if v.varValue is not None:
                if v.varValue > 0:
                    print(v.name)
        for v in model.variables():
            if v.varValue is not None:
                if v.varValue > 0:
                    print(v.varValue)

        # Printausgabe - Binärvariable u[(i)], wenn = 1
        for i in J:
            if value(u[i]) is not None:
                self.logger.debug(f"u({i}) = {u[i]}")

        # Printausgabe - Binärvariable x[(i, j, k)], wenn = 1
        for i in I:
            for j in I:
                for k in range(A):
                    if value(x[(i, j, k)]) == 1:
                        self.logger.debug(f"x({i}, {j}, {k}) = 1.0")

        # Ausfüllen von dict_result_1, wenn = 1
        for i in I:
            for j in I:
                for k in range(A):
                    if value(x[(i, j, k)]) == 1:
                        self.dict_result_1[f'({i}, {j}, {k})'] = value(x[i, j, k])
                        self.x_values[i, j, k] = value(x[i, j, k])

        # Ausfüllen von dict_result_2
        for i in I:
            for j in I:
                for k in range(A):
                    self.dict_result_2[f"x({i},{j},{k})"] = value(x[(i, j, k)])
                    self.x_values_2[i, j, k] = value(x[i, j, k])
        self.logger.debug(self.project_path)
        # with open(self.project_path + "/data/Current_Factory/VRP/result_1.txt", "w") as convert_file:
        #     convert_file.write(json.dumps(self.dict_result_1))
        # with open(self.project_path + "/data/Current_Factory/VRP/result_1.json", "w") as outfile:
        #     json.dump(self.dict_result_1, outfile)
        #
        # with open(self.project_path + "/data/Current_Factory/VRP/result_2.txt", "w") as convert_file:
        #     convert_file.write(json.dumps(self.dict_result_2))
        # with open(self.project_path + "/data/Current_Factory/VRP/result_2.json", "w") as outfile:
        #     json.dump(self.dict_result_2, outfile)
        #
        # with open(self.project_path + "/data/Current_Factory/VRP/x_values.pkl", "wb") as outfile:
        #     pickle.dump(self.x_values, outfile)
        # with open(self.project_path + "/data/Current_Factory/VRP/x_values_2.pkl", "wb") as outfile:
        #     pickle.dump(self.x_values_2, outfile)

        return self.x_values, self.x_values_2


    def get_order_of_transport_new(self):
        dict_order = dict()  # in the dict "dict_order" the order shall be saved as a dict,
        #                      the step number is the key and the routes (saved as (i,j,k)) are the values
        dict_order, remaining_x_values, actual_node_position = self.get_first_step_of_transport()
        step = 1
        # self.logger.debug('#######################################################################################################')
        # self.logger.debug('#######################################################################################################')
        # self.logger.debug('#######################################################################################################')
        # self.logger.debug(f'Verbleibende Kanten gesamt: {remaining_x_values}')
        # self.logger.debug(f'Aktuelle Reihenfolge des Transports: {dict_order}')
        # self.logger.debug(f'Positionen der Fahrzeuge nach Iteration 0: {actual_node_position}')
        # self.logger.debug('#######################################################################################################')
        remaining_x_values_dict_per_step = dict()
        remaining_x_values_dict_per_step[step] = remaining_x_values
        actual_node_position_dict_per_step = dict()
        actual_node_position_dict_per_step[step] = actual_node_position

        if self.get_next_step_of_transport(remaining_x_values_dict_per_step, dict_order,
                                           actual_node_position_dict_per_step, step) == False:
            self.logger.info("no solution possible")
            # print('There is no solution\n')
            return {}
        else:
            self.dict_order_of_transport = dict_order
            self.logger.info("Solution found!")
            # print(f'Solution: {dict_order}')
            return dict_order

    def get_first_step_of_transport(self):
        '''
        This is the ride of the agv to the initial pick-up nodes
        :return:
        '''
        dict_order = dict()  # in the dict "dict_order" the order shall be saved as a dict,
        #                      the step number is the key and the routes (saved as (i,j,k)) are the values
        working_list_order = []  # temporary list, which contains each x(i,j,k) for each step of the order as tuples
        remaining_x_values = self.x_values.copy()  # saves all remaining nodes for the iteration
        actual_node_position = dict()  # saves the temporary node position

        for i, j, k in self.x_values.keys():
            if i == k:
                working_list_order.append((i, j, k))  # adds the paths as a tuple
                remaining_x_values.pop((i, j, k))  # deletes the tuple from the remaining x_values
                actual_node_position[k] = j  # sets the current node position for each vehicle k
            dict_order[0] = working_list_order  # adds the travelled edge to dict

        return dict_order, remaining_x_values, actual_node_position

    def get_next_step_of_transport(self, remaining_x_values_dict_per_step, dict_order,
                                   actual_node_position_dict_per_step, step):
        # If there are no x_values remaining, then a solution is found
        if remaining_x_values_dict_per_step[step] == {}:
            return True

        # Try all combinations possible transports as a candidate from the actual node position of the vehicles
        possible_vehicle_edges = []  # temporary list which contains all the possible edges when vehicle k is at node i
        for _ in range(len(self.factory.agvs)):
            possible_vehicle_edges.append([])

        vehicle_ids = set(vehicle_id for _, _, vehicle_id in self.x_values.keys())
        remaining_visited_edges = dict()
        for k in vehicle_ids:
            remaining_visited_edges[k] = [(i, j) for i, j, vehicle_id in self.x_values.keys() if
                                          self.x_values[(i, j, vehicle_id)] == 1.0 and vehicle_id == k]

        for k in actual_node_position_dict_per_step[step].keys():
            possible_vehicle_edges[k] = [edge for edge in remaining_visited_edges[k] if
                                         edge[0] == actual_node_position_dict_per_step[step][k]]
        # self.logger.debug(f'Mögliche Kanten: {possible_vehicle_edges}')
        num_vehicles = 0
        necessary_vehicles_for_transport = self.count_tuple_combinations(remaining_x_values_dict_per_step[step])
        # self.logger.debug(f'Necessary vehicles for transport: {necessary_vehicles_for_transport}')
        possible_edges = []
        for i in range(self.dimension):  # jeder Knoten i wird durchgegangen
            for k in actual_node_position_dict_per_step[step].values():
                if k == i:
                    num_vehicles += 1  # wird +1, wenn ein Fahrzeug sich am Knoten i befindet
            for key, value in necessary_vehicles_for_transport.items():
                if value <= num_vehicles and i == key[0]:
                    possible_edges.append(key)
            num_vehicles = 0
        # self.logger.debug(f'MÖGLICHE Fahrten gesamt: {possible_edges}')

        for possible_edge in possible_edges:
            working_list_order = []  # temporary list, which contains each x(i,j,k) for each step of the order as tuples
            # self.logger.debug(
            #     '\n#######################################################################################################\n')
            # self.logger.debug(f'Iteration {step}')
            # self.logger.debug(f'Mögliche Kante: {possible_edge}')
            # self.logger.debug(f'Verbleibende Kanten gesamt: {remaining_x_values_dict_per_step[step]}')
            # self.logger.debug(f'Positionen der Fahrzeuge vor Iteration {step}: {actual_node_position_dict_per_step[step]}')
            remaining_x_values_dict_per_step[step + 1] = remaining_x_values_dict_per_step[step].copy()
            actual_node_position_dict_per_step[step + 1] = actual_node_position_dict_per_step[step].copy()
            for i, j, k in remaining_x_values_dict_per_step[step]:
                if i == possible_edge[0] and j == possible_edge[1]:
                    working_list_order.append((i, j, k))  # adds the paths as a tuple
                    remaining_x_values_dict_per_step[step + 1].pop((i, j, k))
                    actual_node_position_dict_per_step[step + 1][
                        k] = j  # sets the current node position for each vehicle k
            dict_order[step] = working_list_order
            len_dict_order = len(dict_order)
            if len(dict_order) > step:
                for i in range(len_dict_order, step):
                    dict_order.pop(i)
            # self.logger.debug(f'Ausgewählte Kante: {possible_edge}')
            # self.logger.debug(f'Verbleibende Kanten gesamt nach Fahrt: {remaining_x_values_dict_per_step[step + 1]}')
            # self.logger.debug(f'Aktuelle Reihenfolge des Transports: {dict_order}')
            # self.logger.debug(f'Positionen der Fahrzeuge nach Iteration {step}: {actual_node_position_dict_per_step[step + 1]}')
            # self.logger.debug(
            #     '\n#######################################################################################################\n')
            if self.get_next_step_of_transport(remaining_x_values_dict_per_step, dict_order,
                                               actual_node_position_dict_per_step, step + 1) is True:
                return True

        return False

    def count_tuple_combinations(self, data_dict):
        count_dict = {}
        # Iteriere über jeden Schlüssel im Dictionary
        for key in data_dict.keys():
            # Extrahiere die ersten beiden Elemente des Schlüssel-Tupels
            pair = (key[0], key[1])
            if pair in count_dict:
                # Zähle jedes Auftreten des Paares
                count_dict[pair] += 1
            else:
                count_dict[pair] = 1
        return count_dict

    def get_machine_idle_process_times(self):
        machine_idle_process_times = self.factory_simulation.get_machine_idle_process_times()
        # print(f'Machine Idle Process Times = {machine_idle_process_times}')
        time_matrix = np.zeros(shape=(self.amount_of_nodes, self.amount_of_nodes))
        list_of_factory_objects_name = self.create_list_of_factory_objects()
        m = 1
        for i in range(self.dimension):
            for j in range(self.dimension):
                if (type(self.list_of_factory_objects_input_output[i]) == Machine or
                        type(self.list_of_factory_objects_input_output[i]) == Warehouse):
                    if self.list_agv_input_output[i][0] == 'output':
                        if (type(self.list_of_factory_objects_input_output[j]) == Machine or
                                type(self.list_of_factory_objects_input_output[j]) == Warehouse):
                            if self.list_agv_input_output[j][0] == 'input':
                                if self.list_agv_input_output[i][1] == self.list_agv_input_output[j][1]:  # productcheck
                                    if self.list_of_factory_objects_input_output[i].has_product(
                                            self.list_agv_input_output[i][1]):
                                        # print(
                                        #     '---------------------------------------------------------------------------')
                                        # print(f'{m:03d}. Lieferbeziehung gefunden! \n'
                                        #                   f'Lieferung von {self.list_of_factory_objects_input_output[i].name} '
                                        #                   f'(Knoten {i}, TYPE: {type(self.list_of_factory_objects_input_output[i])}) nach {self.list_of_factory_objects_input_output[j].name} '
                                        #                   f'(Knoten {j}, TYPE: {type(self.list_of_factory_objects_input_output[j])}) \n'
                                        #                   f'Geliefertes Produkt: {self.list_agv_input_output[i][1]}')
                                        if (type(self.list_of_factory_objects_input_output[j]) == Machine and (
                                                self.list_of_factory_objects_input_output[j].get_buffer_status()[0] != 0
                                                or self.list_of_factory_objects_input_output[j].status == 'idle')):
                                            # Kann nur beliefert werden, wenn der Buffer von Machine[j] frei ist
                                            time_matrix[i][j] = machine_idle_process_times[self.list_of_factory_objects_input_output[j].name]['idle']
                                            # if self.list_of_factory_objects_input_output[j].get_status() == 'process':
                                            #     time_matrix[i][j] = machine_idle_process_times[self.list_of_factory_objects_input_output[j].name]['idle'] - self.list_of_factory_objects_input_output[j].rest_process_time
                                            # else:
                                            #     time_matrix[i][j] = machine_idle_process_times[
                                            #         self.list_of_factory_objects_input_output[j].name]['idle']
                                            # print(machine_idle_process_times[self.list_of_factory_objects_input_output[j].name])
                                            # print(machine_idle_process_times[self.list_of_factory_objects_input_output[j].name]['idle'])
                                            # print(self.list_of_factory_objects_input_output[j].rest_process_time)
                                        elif type(self.list_of_factory_objects_input_output[j]) == Warehouse:
                                            time_matrix[i][j] = 0.1
                                        if (self.check_amount_of_agvs_for_transport_1_4_6_configuration(
                                                self.factory.product_types[self.list_agv_input_output[i][1]]) >
                                                len(self.factory.get_num_free_agv_for_vrp())):
                                            time_matrix[i][j] = 0
                                            # self.delivery_matrix_with_agv[i][j] = delivery_matrix_with_agv[i][j]
                                        # else:
                                        #     delivery_matrix_with_agv[i][
                                        #         j] = self.check_amount_of_agvs_for_transport_1_4_6_configuration(
                                        #         self.factory.product_types[output_product])
                                        m += 1

        time_matrix_pd = pd.DataFrame(time_matrix, columns=list_of_factory_objects_name,
                                          index=list_of_factory_objects_name)
        if self.write_factory_files:
            time_matrix_pd.to_csv(
                self.project_path + '/data/Current_Factory/VRP/time_matrix.csv', sep=';')
        time_matrix.tolist()
        # print(f'Time Matrix = {time_matrix}')
        return time_matrix.tolist()

    def get_route_for_vehicle(self, dict_order_of_transport, vehicle_id):
        visited_edges = []
        # Durchlaufen aller Schritte in den Transporten
        for step in sorted(dict_order_of_transport.keys()):
            # Durchlaufen aller Routen in diesem Schritt
            for route in dict_order_of_transport[step]:
                # Extrahieren der Fahrzeug-ID aus dem Tupel und Vergleich
                if route[2] == vehicle_id:
                    # Hinzufügen des Tupels (i, j) zur Liste der besuchten Kanten
                    visited_edges.append((route[0], route[1]))
        return visited_edges

    def write_state(self, D, Q, A):
        """
        Writes the complete input of te VRP as concatenated 1D np.array in a file.

        Args:
            D: Distance Marix as 2d Np.array
            Q: Needed AGV for delivery matrix as 2d Np.array
            A: Amount of free AGVs

        Returns: None

        """
        D = np.array(D)
        Q = np.array(Q)
        state = D.reshape(1, -1)
        state = np.concatenate((state, Q.reshape(1, -1)), axis=1)
        A = np.array([A]).reshape((1,-1))
        state = np.concatenate((state, A), axis=1)
        self.logger.debug(f"state_shape: {state.shape}")
        # np.savetxt(self.project_path + f"/data/Current_Factory/Imitation_data/state_{self.step_counter}.csv", state,
        #            delimiter=",")

        with open(self.project_path + "/data/Current_Factory/Imitation_data/state.csv", "a", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(state[0].tolist())
    
    def write_output(self, routing, n_agv_free):
        """
        Writes the first step form the ordered routing in a file as 1-d np.array of the following hierarchy:

            source_obj, target_obj, agv_number

        the values are coming from self.list_of_factory_objects_input_output

        Args:
            routing: Dict the routing from the ordered VRP Solution

        Returns: None

        """
        solution_found = len(routing.keys()) > 1
        n_objects = len(self.factory.agvs) + 2 * (len(self.factory.warehouses) + len(self.factory.machines))
        self.logger.debug(f"OBJECTS: {n_objects}")
        n_agv = len(self.factory.agvs)
        self.logger.debug(f"AGVS: {n_agv}")
        output = np.zeros(n_objects*n_objects*n_agv)
        if solution_found:
            step = routing[1]
            for route in step:
                source = route[0] + (n_agv - n_agv_free)
                target = route[1] + (n_agv - n_agv_free)
                agv = self.factory.agvs.index(self.list_of_factory_objects_input_output[route[2]])
                target_index = source*(n_objects*n_agv) + target * n_agv + agv
                output[target_index] = 1

        output = output.reshape(1, -1)

        with open(self.project_path + "/data/Current_Factory/Imitation_data/output.csv", "a", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(output[0].tolist())

    def write_observation(self):
        obs = self._create_observation()
        with open(self.project_path + "/data/Current_Factory/Imitation_data/observation.csv", "a", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(obs)


    def _create_observation(self):
        observation = []
        for agv in self.factory.agvs:
            if agv.is_free:
                observation.append(1.0)
            else:
                observation.append(0.0)
            if agv.loaded_product is None:
                observation.append(1.0)
            else:
                observation.append(0.0)
            observation += self._get_agv_distances_normal(agv)
        for warehouse in self.factory.warehouses:
            if len(warehouse.buffer_output_load) > 0:
                observation.append(1.0)
            else:
                observation.append(0.0)
            observation.append(warehouse.get_production_rest_time_percent())
        for machine in self.factory.machines:
            input_priority, output_priority = machine.get_buffer_status()
            observation.append(input_priority*0.25)
            observation.append(output_priority*0.25)
            observation.append(machine.get_production_rest_time_percent())
        return observation

    def _get_agv_distances_normal(self, agv):
        agv_machines_distances_normal = []
        pos_agv = agv.get_middle_position()
        for warehouse in self.factory.warehouses:
            pos_warehouse = warehouse.get_middle_position()
            agv_machines_distances_normal.append(self._calculate_agv_distances(pos_agv, pos_warehouse))

        for machine in self.factory.machines:
            pos_machine = machine.get_middle_position()
            agv_machines_distances_normal.append(self._calculate_agv_distances(pos_agv, pos_machine))

        max_distance = max(agv_machines_distances_normal)
        for i in range(len(agv_machines_distances_normal)):
            agv_machines_distances_normal[i] = agv_machines_distances_normal[i] / max_distance
        return agv_machines_distances_normal

    @staticmethod
    def _calculate_agv_distances(pos_a, pos_b):
        return math.sqrt(math.pow(pos_a[0] - pos_b[0], 2) + math.pow(pos_a[1] - pos_b[1], 2))

    def get_dynamic_routing(self, step_counter):
        if self.use_paths:
            D = self.factory.path_graph.get_object_distance_matrix(only_free_agv=True)
        else:
            D = self.get_distance_matrix()


        list_of_factory_objects_name = self.create_list_of_factory_objects()
        distance_matrix_pd = pd.DataFrame(D, columns=list_of_factory_objects_name,
                                          index=list_of_factory_objects_name)
        if self.write_factory_files:
            distance_matrix_pd.to_csv(self.project_path + f'/data/Current_Factory/VRP/distance_matrix_{step_counter}.csv', sep=';')

        Q = self.get_amount_of_agv_for_delivery_as_matrix_1_4_6_configuration()  # Konfiguration der FTF
        A = self.factory.get_num_free_agv_for_vrp()
        T = self.get_machine_idle_process_times()

        self.cVRPPDmDnR(num_AGVs=len(A), D=D, Q=Q, T=T)
        routing = self.get_order_of_transport_new()

        if self.write_state_output:
            self.write_state(D, Q, len(A))
            self.write_observation()
            self.write_output(routing, len(A))

        return routing


def create_default_factory_5(self):
    """
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
    """
    self.length = 8
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

    self.warehouses.append(Warehouse())
    self.warehouses[0].pos_x = 2
    self.warehouses[0].pos_y = 3
    self.warehouses[0].length = 1
    self.warehouses[0].width = 2
    self.warehouses[0].pos_input = [2, 4]
    self.warehouses[0].pos_output = [2, 3]
    self.warehouses[0].input_products = ['empty_w1']
    self.warehouses[0].output_products = ['lager_m1', 'lager_m2', 'lager_m3']
    self.warehouses[0].factory = self

    self.machines.append(Machine())
    self.machines[0].name = 'M1'
    self.machines[0].pos_x = 5
    self.machines[0].pos_y = 0
    self.machines[0].length = 1
    self.machines[0].width = 2
    self.machines[0].pos_input = [5, 1]
    self.machines[0].pos_output = [5, 0]
    self.machines[0].input_products = ['lager_m1']
    self.machines[0].output_products = ['m1_m3']
    self.machines[0].factory = self

    self.machines.append(Machine())
    self.machines[1].name = 'M2'
    self.machines[1].pos_x = 5
    self.machines[1].pos_y = 3
    self.machines[1].length = 1
    self.machines[1].width = 2
    self.machines[1].pos_input = [5, 4]
    self.machines[1].pos_output = [5, 3]
    self.machines[1].input_products = ['lager_m2', 'm3_m2']
    self.machines[1].output_products = ['empty']
    self.machines[1].factory = self

    self.machines.append(Machine())
    self.machines[2].name = 'M3'
    self.machines[2].pos_x = 6
    self.machines[2].pos_y = 8
    self.machines[2].length = 1
    self.machines[2].width = 2
    self.machines[2].pos_input = [6, 9]
    self.machines[2].pos_output = [6, 8]
    self.machines[2].input_products = ['lager_m3', 'm1_m3']
    self.machines[2].output_products = ['m3_lager']
    self.machines[2].factory = self

    for i in range(6):
        self.loading_stations.append(LoadingStation())
        self.loading_stations[i].name = f'Loading_Station_{i}'
        self.loading_stations[i].pos_x = 0
        self.loading_stations[i].pos_y = i
        self.loading_stations[i].length = 1
        self.loading_stations[i].width = 1
        self.agvs.append(AGV([0, i]))
        self.agvs[i].name = i
        self.agvs[i].thread_running = False
        self.agvs[i].factory = self

    self.fill_grid()


def create_default_factory_5_running(self):
    """
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
    """
    self.length = 8
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

    self.warehouses.append(Warehouse())
    self.warehouses[0].pos_x = 2
    self.warehouses[0].pos_y = 3
    self.warehouses[0].length = 1
    self.warehouses[0].width = 2
    self.warehouses[0].pos_input = [2, 4]
    self.warehouses[0].pos_output = [2, 3]
    self.warehouses[0].input_products = ['empty_w1']
    self.warehouses[0].output_products = ['lager_m1', 'lager_m2', 'lager_m3']
    self.warehouses[0].buffer_output = [30]
    self.warehouses[0].factory = self
    self.warehouses[0].buffer_output_load = [Product(), Product(), Product()]
    self.warehouses[0].buffer_output_load[0].name = 'lager_m2'
    self.warehouses[0].buffer_output_load[1].name = 'lager_m1'
    self.warehouses[0].buffer_output_load[2].name = 'lager_m3'
    self.warehouses[0].buffer_output_load[0].length = 1200
    self.warehouses[0].buffer_output_load[0].width = 1000
    self.warehouses[0].buffer_output_load[1].length = 500
    self.warehouses[0].buffer_output_load[1].width = 500
    self.warehouses[0].buffer_output_load[2].length = 1000
    self.warehouses[0].buffer_output_load[2].width = 1000

    self.machines.append(Machine())
    self.machines[0].name = 'M1'
    self.machines[0].pos_x = 5
    self.machines[0].pos_y = 0
    self.machines[0].length = 1
    self.machines[0].width = 2
    self.machines[0].pos_input = [5, 1]
    self.machines[0].pos_output = [5, 0]
    self.machines[0].input_products = ['lager_m1']
    self.machines[0].output_products = ['m1_m3']
    self.machines[0].factory = self
    self.machines[0].buffer_output_load = [Product()]
    self.machines[0].buffer_output_load[0].name = 'm1_m3'
    self.machines[0].buffer_output_load[0].length = 1200
    self.machines[0].buffer_output_load[0].width = 1000

    self.machines.append(Machine())
    self.machines[1].name = 'M2'
    self.machines[1].pos_x = 5
    self.machines[1].pos_y = 3
    self.machines[1].length = 1
    self.machines[1].width = 2
    self.machines[1].pos_input = [5, 4]
    self.machines[1].pos_output = [5, 3]
    self.machines[1].input_products = ['lager_m2', 'm3_m2']
    self.machines[1].output_products = [
        'default_product_1']  # todo hier vorher "empty, war jedoch nicht in product_types drin --> change product hat nicht funktioniert
    self.machines[1].factory = self
    self.machines[1].buffer_output_load = [Product()]

    self.machines.append(Machine())
    self.machines[2].name = 'M3'
    self.machines[2].pos_x = 6
    self.machines[2].pos_y = 8
    self.machines[2].length = 1
    self.machines[2].width = 2
    self.machines[2].pos_input = [6, 9]
    self.machines[2].pos_output = [6, 8]
    self.machines[2].input_products = ['lager_m3', 'm1_m3']
    self.machines[2].output_products = ['default_product_1']  # todo vorher empty
    self.machines[2].factory = self
    self.machines[2].buffer_output_load = [Product()]
    self.machines[2].buffer_output_load[0].name = 'm3_lager'
    self.machines[2].buffer_output_load[0].length = 1500
    self.machines[2].buffer_output_load[0].width = 800

    for i in range(6):
        self.loading_stations.append(LoadingStation())
        self.loading_stations[i].name = f'Loading_Station_{i}'
        self.loading_stations[i].pos_x = 0
        self.loading_stations[i].pos_y = i
        self.loading_stations[i].length = 1
        self.loading_stations[i].width = 1
        self.agvs.append(AGV([0, i]))
        self.agvs[i].name = i
        self.agvs[i].thread_running = True
        self.agvs[i].factory = self

    self.fill_grid()


"""
Notizen:
- Transportkostensätze darstellen durch Matrixmultiplikation (Menge AGV * Distanz)
- Zeitabhängigkeit reinbringen: immer wenn ein Produkt neu angeboten/nachgefragt wird, Modell neu durchrechnen?!
"""
