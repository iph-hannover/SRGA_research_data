import datetime
import os.path

import numpy as np
import pandas as pd
import pickle
import pygame
import time
import pytups as pt
from math import sqrt, ceil
from pulp import *
from itertools import chain, combinations
from copy import deepcopy

from FactoryObjects.Factory import Factory
from FactoryObjects.Warehouse import Warehouse
from FactoryObjects.Machine import Machine
from FactoryObjects.Product import Product
from FactoryObjects.LoadingStation import LoadingStation
from FactoryObjects.AGV import AGV
from FactoryObjects.Path import Path

import VRP_Simulation.random_factory_generator
import VRP_Simulation.random_grid_factory_with_paths
from GAMS_Schnittstelle import table_to_gams

class VRP_cellAGV():
    """
    This class contains all the information for the VRP.
    """

    def __init__(self, factory: Factory):
        self.factory: Factory = factory
        self.agv = AGV()
        self.agv.thread_running = False
        self.agv.length = 500  # length of agv in mm
        self.agv.width = 500  # width of agv in mm
        self.amount_of_nodes = self.factory.get_amount_of_delivery_nodes()
        self.dimension = self.amount_of_nodes
        self.list_of_factory_objects_input_output = self.factory.get_list_of_factory_objects_agvs_warehouse_machines_input_output()
        self.list_agv_input_output = self.factory.get_list_input_output_product()

        self.distance_matrix = np.zeros(shape=(self.amount_of_nodes, self.amount_of_nodes))
        self.distance_matrix = self.get_distance_matrix()

        self.delivery_matrix = np.zeros(shape=(self.amount_of_nodes, self.amount_of_nodes))
        self.delivery_matrix = self.get_delivery_relationship()
        self.delivery_matrix_with_agv = np.zeros(shape=(self.amount_of_nodes, self.amount_of_nodes))
        self.delivery_matrix_with_agv = self.get_amount_of_agv_for_delivery_as_matrix_1_4_6_configuration()
        # Dictionaries für das Speichern der Werte der Binärvariablen in einer Datei
        self.dict_result_1 = dict()
        self.dict_result_2 = dict()
        self.x_values = dict()
        self.x_values_2 = dict()
        self.dict_order_of_transport = dict()
        # Values for Visualisation with pygame
        self.pixel_size = 25
        self.height = self.factory.width
        self.width = self.factory.length
        self.reference_size = self.factory.cell_size * 1000
        self.screen = None

        self.fitness = None
        self.duration = None
        self.mip_gap = None
        self.routing = None

    def create_list_of_factory_objects(self):
        """
        This function writes all factory objects (Warehouses, Machines, Loading stations) to a list.
        :return: list_of_factory_objects_name
                 A list of all factory objects in the order mentioned above as a list
        """
        list_of_factory_objects_name = []
        j = 0
        for i in self.list_agv_input_output:
            if isinstance(self.list_agv_input_output[j][2], AGV):
                list_of_factory_objects_name.append(self.list_agv_input_output[j][2].name)
            else:
                list_of_factory_objects_name.append(f"{self.list_agv_input_output[j][2].name}_{self.list_agv_input_output[j][0]}")
            j += 1

        # for i in range(self.dimension):
        #     if isinstance(self.list_of_factory_objects_input_output[i], AGV):
        #         list_of_factory_objects_name.append(self.list_of_factory_objects_input_output[i].name)
        #     else:
        #         if self.list_of_factory_objects_input_output[i] != self.list_of_factory_objects_input_output[i - 1]:
        #             list_of_factory_objects_name.append(f'{self.list_of_factory_objects_input_output[i].name}_input')
        #         else:
        #             list_of_factory_objects_name.append(f'{self.list_of_factory_objects_input_output[i].name}_output')
        # return list_of_factory_objects_name

    def create_dataframe_of_factory_objects(self):
        """
        This function creates a Pandas data frame for all factory objects from the list of all factory objects.
        :return: list_of_factory_objects_pd
                 The data frame with the factory objects.
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
        list_of_factory_objects_pd = pd.DataFrame(list_of_factory_objects_name)
        list_of_factory_objects_pd['ID'] = list_of_factory_objects_pd.index + 1
        # print(list_of_factory_objects_pd)
        return list_of_factory_objects_pd

    def create_file_for_list_of_factory_objects(self):
        """
        This function creates a CSV file from the Pandas data frame.
        The file is saved under the following path: data/Current_Factory/VRP/list_of_factory_objects.csv
        :return: None
        """
        list_of_factory_objects_pd = pd.DataFrame(self.create_list_of_factory_objects())
        # list_of_factory_objects_pd.to_csv('C:/code/ZellFTF_2DSim/data/Current_Factory/VRP/list_of_factory_objects.csv', sep=';')

    def get_distance_matrix(self):
        """
        Creates a matrix with the distance relationships between individual factory objects I and J.
        :return: distance_matrix as a 2D-np-array
        """
        # Die Matrix self.distance_matrix enthält die Distanzen aller Maschinen, Warenhäuser und AGVs.
        # Aus Gründen der Vereinfachung wird die Manhattan-Distanz verwendet. Mögliche Kollisionen werden ignoriert.
        distance_matrix = np.zeros(shape=(self.amount_of_nodes, self.amount_of_nodes))
        list_of_factory_objects_name = self.create_list_of_factory_objects()
        print(list_of_factory_objects_name)
        distance_x = 0
        distance_y = 0
        for i in range(self.dimension):
            for j in range(self.dimension):
                # 1st if: AFV - AGV
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
                # self.distance_matrix[i][j] = distance_x + distance_y
                # distance_matrix[i][j] = distance_x + distance_y
                # Euklid-Distance
                self.distance_matrix[i][j] = sqrt((distance_x * distance_x) + (distance_y * distance_y)) + 0.01
                distance_matrix[i][j] = sqrt((distance_x * distance_x) + (distance_y * distance_y)) + 0.01
                # print(
                #     f'i = {i:02d} | j = {j:02d} --- X-Distanz = {distance_x:03d} | Y-Distanz = {distance_y:03d} --- '
                #     f'Distanz ={self.distance_matrix[i][j]}')
        distance_matrix_pd = pd.DataFrame(distance_matrix, columns=list_of_factory_objects_name,
                                          index=list_of_factory_objects_name)
        # distance_matrix_pd.to_csv('C:/code/ZellFTF_2DSim/data/Current_Factory/VRP/distance_matrix.csv', sep=';')
        print(distance_matrix.tolist())
        return distance_matrix.tolist()

    def get_delivery_relationship(self):
        '''
        Creates a matrix with the delivery relationships between individual factory objects I and J.
        If a delivery from I to J takes place, a '1' is set at the corresponding position in the matrix.
        The matrix is saved as a CSV file under the path: data/Current_Factory/VRP/delivery_matrix.csv
        :return: delivery_matrix as a 2D-np-array
        '''
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
                                    # print('---------------------------------------------------------------------------')
                                    # print(f'{m}. Lieferbeziehung gefunden!')
                                    # print(f'Lieferung von {list_of_factory_objects[i].name} nach '
                                    #       f'{list_of_factory_objects[j].name}, Geliefertes Produkt: {input_product}')
                                    delivery_matrix[i][j] = 1
                                    self.delivery_matrix[i][j] = delivery_matrix[i][j]
                                    m += 1
        delivery_matrix_pd = pd.DataFrame(delivery_matrix, columns=list_of_factory_objects_name,
                                          index=list_of_factory_objects_name)
        # delivery_matrix_pd.to_csv('C:/code/ZellFTF_2DSim/data/Current_Factory/VRP/delivery_matrix.csv', sep=';')
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
                                    print(
                                        '---------------------------------------------------------------------------')
                                    print(f'{m:03d}. Lieferbeziehung gefunden! \n'
                                          f'Lieferung von {self.list_of_factory_objects_input_output[i].name} '
                                          f'(Knoten {i}) nach {self.list_of_factory_objects_input_output[j].name} '
                                          f'(Knoten {j}) \n'
                                          f'Geliefertes Produkt: {self.list_agv_input_output[i][1]}')
                                    delivery_matrix_with_agv[i][j] = (
                                        self.check_amount_of_agvs_for_transport_free_configuration(
                                            self.factory.product_types[self.list_agv_input_output[i][1]]))
                                    self.delivery_matrix_with_agv[i][j] = delivery_matrix_with_agv[i][j]
                                    m += 1
        delivery_matrix_with_agv_pd = pd.DataFrame(delivery_matrix_with_agv, columns=list_of_factory_objects_name,
                                                   index=list_of_factory_objects_name)
        delivery_matrix_with_agv_pd.to_csv('data/Current_Factory/VRP/delivery_matrix_with_agv.csv', sep=';')
        print(type(delivery_matrix_with_agv))
        print(delivery_matrix_with_agv.tolist())
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
        agv_in_a_row = ceil(length_ratio)
        agv_side_by_side = ceil(width_ratio)
        amount_of_agv = agv_in_a_row * agv_side_by_side
        print('AGV hintereinander: {}'.format(agv_in_a_row))
        print('AGV nebeneinander : {}'.format(agv_side_by_side))
        print('AGV gesamt        : {}'.format(amount_of_agv))
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
                                if self.list_agv_input_output[i][1] == self.list_agv_input_output[j][1]:
                                    print(
                                        '---------------------------------------------------------------------------')
                                    print(f'{m:03d}. Lieferbeziehung gefunden! \n'
                                          f'Lieferung von {self.list_of_factory_objects_input_output[i].name} '
                                          f'(Knoten {i}) nach {self.list_of_factory_objects_input_output[j].name} '
                                          f'(Knoten {j}) \n'
                                          f'Geliefertes Produkt: {self.list_agv_input_output[i][1]}')
                                    delivery_matrix_with_agv[i][j] = (
                                        self.check_amount_of_agvs_for_transport_1_4_6_configuration(
                                        self.factory.product_types[self.list_agv_input_output[i][1]]))
                                    self.delivery_matrix_with_agv[i][j] = delivery_matrix_with_agv[i][j]
                                    m += 1
        delivery_matrix_with_agv_pd = pd.DataFrame(delivery_matrix_with_agv, columns=list_of_factory_objects_name,
                                                   index=list_of_factory_objects_name)
        # delivery_matrix_with_agv_pd.to_csv('C:/code/ZellFTF_2DSim/data/Current_Factory/VRP/delivery_matrix_with_agv.csv', sep=';')
        # print("DELIVERY MATRIX:")
        # print(type(delivery_matrix_with_agv))
        # print(delivery_matrix_with_agv.tolist())
        return delivery_matrix_with_agv.tolist()

    def check_amount_of_agvs_for_transport_1_4_6_configuration(self, product):
        """
        Calculates the amount of agv, which is necessary to transport a product from I to J.
        Depending on the length of the product, it is determined how many AGVs have to travel one behind the other.
        Depending on the width of the product, the number of AGVs that must travel side by side is determined.
        :param product: product which is transported from I to J
        :return: amount_of_agv as an int
        """
        # print('Länge des Produkts : {}'.format(product['length']))
        # print('Breite des Produkts: {}'.format(product['width']))
        # print('Länge des AGV : {}'.format(self.agv.length / 1000))
        # print('Breite des AGV: {}'.format(self.agv.width / 1000))
        length_of_product = product['length']
        width_of_product = product['width']
        length_of_agv = self.agv.length  # / 1000
        width_of_agv = self.agv.width  # / 1000
        length_ratio = length_of_product / length_of_agv
        width_ratio = width_of_product / width_of_agv
        # print('Längenverhältnis : {}'.format(length_ratio))
        # print('Breitenverhältnis: {}'.format(width_ratio))
        if length_ratio <= 1:
            if width_ratio <= 1:
                agv_in_a_row = 1
                agv_side_by_side = 1
                amount_of_agv = agv_in_a_row * agv_side_by_side
                print('AGV hintereinander: {}'.format(agv_in_a_row))
                print('AGV nebeneinander : {}'.format(agv_side_by_side))
                print('AGV gesamt        : {}'.format(amount_of_agv))
                return amount_of_agv
        elif length_ratio <= 2:
            if width_ratio <= 2:
                agv_in_a_row = 2
                agv_side_by_side = 2
                amount_of_agv = agv_in_a_row * agv_side_by_side
                print('AGV hintereinander: {}'.format(agv_in_a_row))
                print('AGV nebeneinander : {}'.format(agv_side_by_side))
                print('AGV gesamt        : {}'.format(amount_of_agv))
                return amount_of_agv
        elif length_ratio <= 3:
            if width_ratio <= 2:
                agv_in_a_row = 3
                agv_side_by_side = 2
                amount_of_agv = agv_in_a_row * agv_side_by_side
                print('AGV hintereinander: {}'.format(agv_in_a_row))
                print('AGV nebeneinander : {}'.format(agv_side_by_side))
                print('AGV gesamt        : {}'.format(amount_of_agv))
                return amount_of_agv
        else:
            print('Transport nicht möglich')
            return None

        agv_in_a_row = ceil(length_ratio)
        agv_side_by_side = ceil(width_ratio)
        amount_of_agv = agv_in_a_row * agv_side_by_side
        print('AGV hintereinander: {}'.format(agv_in_a_row))
        print('AGV nebeneinander : {}'.format(agv_side_by_side))
        print('AGV gesamt        : {}'.format(amount_of_agv))
        return amount_of_agv

    def cVRPPDmDnR(self, num_AGVs, D, Q, machine_times):

        starttime = time()

        ###################################
        ###  Eingabegrößen des Modells  ###
        ###################################

        K = []  # Anzahl der Depots, wo die einzelnen FTF stehen - jedes FTF hat ein Depot
        I = []  # Orte/Knoten, die besucht werden (inkl. Depots, Abhol- und Lieferpunkten), wird aus der Distanzmatrix berechnet
        J = []  # Orte/Knoten der Jobs (inkl. Abhol- und Lieferpunkten)
        A = num_AGVs  # Anzahl der zur Verfügung stehenden Fahrzeuge
        v = 1
        Q_counter = 0  # Zähler, der alle Routen z#hlt, für die mehr als 1 Fahrzeug benötigt wird.
        M = 1000000  # Große Zahl für oder Vergleich Startbedingung mit Q_counter
        QQ = []  # Abholknoten
        QL = []
        w = 1

        for i in range(A):
            K.append(i)
        for i in range(len(D)):
            I.append(i)
        num_locations = len(D)
        J = list(set(I).difference(K))

        for j in J:
            for l in J:
                if Q[j][l] > 0:
                    QQ.append(j)
                    QL.append(l)
                    if Q[j][l] > 0:
                        Q_counter += 1

        if type(D).__module__ == 'numpy':
            D = D.tolist()

        print(f"K = {K}")
        print(f"I = {I}")
        print(f"J = {J}")
        print(f"D = {D}")
        print(f"Q = {Q}")
        solver_list = listSolvers(onlyAvailable=True)
        print(f"Verfügbare Solver: {solver_list}")
        # print(f"Q_Counter = {Q_counter}")
        print(f"QQ = {QQ}")

        ##################################
        ###  Formulierung des Modells  ###
        ##################################

        ###################
        # Initialisierung #
        ###################

        # cellular VRP with Pickups and Delivery and a single Depot
        model = LpProblem("cVRPPDmDnR", LpMinimize)

        x = LpVariable.dicts("x", [(i, j, k) for i in I for j in I for k in K], 0, 1, LpBinary)
        # x_(i, j, k) ist eine Binärvariable, die 1 ist, wenn das Transportfahrzeug k von Ort i direkt zu Ort j fährt; sonst 0.
        # b = LpVariable.dicts("b", [i for i in QQ], 0, 1, LpBinary)
        # s = LpVariable.dicts("s", [(i, j, k) for i in I for j in I for k in K if Q[i][j] > 0], 0, len(QQ))
        # c = LpVariable.dicts("c", [(j, steps) for j in QQ for steps in range(Q_counter)], 0, 1, LpBinary)
        u = LpVariable.dicts("u", [i for i in J], 1, len(J), LpInteger)
        # u_(i) gibt die Reihenfolge an, in welcher die Knoten i besucht werden
        # (FUNKTIONIERT NICHT, SOFERN EIN KNOTEN MEHRFACH ANGEFAHREN WIRD!!!)

        # original
        arrival_time = LpVariable.dicts("arrival_time", [(i, j, k) for i in I for j in I for k in K], 0, None,
                                        LpContinuous)
        # arrival_time_(j, k) gibt die Zeit an, zu welcher das Fahrzeug k spätestens am Knoten j angekommen ist
        max_arrival_time = LpVariable.dicts("max_arrival_time", [(i, j) for i in I for j in I], 0, None, LpContinuous)
        # Die max_arrival_time_(j) ist die Zeit des Fahrzeugs, welches zuletzt am Knoten j ankommt
        wait_time = LpVariable.dicts("wait_time", [(i, j, k) for i in I for j in I for k in K], 0, None, LpContinuous)
        # Die wait_time_(j, k) ist die Zeit, die ein Fahrzeug k maximal auf andere Fahrzeuge am Knoten k wartet.
        max_total_arrival_time = LpVariable("transport_duration", 0, None, LpContinuous)
        # Die max_total_arrival_time ist die Dauer, die der gesamte Transport in Anspruch nimmt.

        # idle_time = LpVariable.dicts("idle_time", [(i, j, k) for i in I for j in QL for k in K], 0, None,
        #                             LpContinuous)
        # # Die idle_time_(i, j, k) gibt die Zeit an, die eine Maschine bei Ankunft eines Fahrzeugs im idle-Zustand sein wird
        # max_idle_time = LpVariable.dicts("max_idle_time", [(i, j) for i in I for j in I], 0, None, LpContinuous)
        # # Die max_idle_time_(j) ist die Zeit des Fahrzeugs, welches zuletzt am Knoten j ankommt

        # print(x)
        # print(b)
        # print(s)
        # print(c)
        # print(u)

        ################
        # Zielfunktion #
        ################

        # Minimierung der gesamten zurückgelegten Distanz
        # model += lpSum(D[i][j] * x[(i, j, k)] for i in I for j in I for k in K), "Minimize_Total_Distance"

        # Minimierung der Transportdauer
        model += max_total_arrival_time, "Minimize_Transport_Duration"

        # model += w * (lpSum(D[i][j] * x[(i, j, k)] for i in I for j in I for k in K)) + (1 - w) * max_total_arrival_time

        # Minimerung der Verspätungen der Lieferungen

        ####################
        # Nebenbedingungen #
        ####################

        ####################################
        # Depots
        # Fahrzeug k startet im Depot h == k
        for k in range(A):
            model += (lpSum(x[(k, j, k)] for j in I) == 1, f'2_Start_im_Depot_{k}_Fahrzeug_{k}')
            model += ((lpSum(x[(k, k, k)]) + lpSum(x[(k, j, k)] for j in QQ)) == 1,
                      f'3_Entweder_bleibt_{k}_im Depot_oder_faehrt_zu_{QQ}')

        # Von jedem Depot kann am Anfang maximal ein Fahrzeug starten.
        for i in K:
            model += (lpSum(x[(i, j, k)] for j in I for k in K) <= 1, f'4_Anfangszustand_Depot_{i}')

        # Fahrzeuge dürfen nicht von Abholknoten zu Abholknoten fahren
        for k in range(A):
            for i in QQ:
                for j in QQ:
                    # if i != j:
                    model += x[(i, j, k)] == 0, f"5_Verbot_Fahrzeug_{k}_von_Abholknoten_{i}_zu_Abholknoten_{j}"

        # Fahrzeuge dürfen nicht von Lieferknoten zu Lieferknoten fahren
        for k in range(A):
            for i in QL:
                for j in QL:
                    # if i != j:
                    model += x[(i, j, k)] == 0, f"6_Verbot_Fahrzeug_{k}_von_Lieferknoten_{i}_zu_Lieferknoten_{j}"

        ################################################################################################################
        # Reihenfolge des Transports zur Verhinderung von Deadlocks
        for k in K:
            for i in J:
                for j in J:
                    if i != j:
                        model += u[i] - u[j] + (len(J) * x[(i, j, k)]) <= len(
                            J) - 1, f"7_MTZ_Subtour_{i}_{j}_Fahrzeug_{k}"

        ################################################################################################################
        # Flusskonservierung - Fluss von Fahrzeugen in den Knoten muss größer oder gleich dem Fluss von Fahrzeugen
        #                      aus dem Knoten sein - außer für das Depot. Dadurch müssen Fahrzeuge nicht ins Depot zurück.
        for j in J:
            for k in range(A):
                model += (lpSum(x[(i, j, k)] for i in I) - lpSum(x[(j, h, k)] for h in I) >= 0,
                          f"8_Flusskonservierung_Knoten_{j}_Fahrzeug_{k}")

        ####################################################################################################################
        # Jeder Lieferung müssen die richtige Anzahl an AGV zugeordnet werden.
        for i in J:
            for j in J:
                if i != j:
                    if Q[i][j] > 0:
                        model += (lpSum(x[(i, j, k)] for k in K) == Q[i][j],
                                  f'9_Anzahl_AGV_pro_Lieferung_Knoten_{i}_nach_Knoten_{j}')

        # Zu jedem Abholknoten sollen nur so viele Fahrzeuge fahren, wie für die Lieferung benötigt werden
        for i in J:
            for j in J:
                if i != j:
                    if Q[i][j] > 0:
                        # print("i =", i, "j =", j, "Benötigte Fahrzeuge =", Q[i][j])
                        model += lpSum(x[(h, i, k)] for h in I for k in K) == Q[i][
                            j], f'10_Anzahl_AGV_zu_Abholknoten_{i}'

        ####################################################################
        # Die Anzahl an AGVs, die die Depots verlassen und zu den Abholorten fahren, muss passen.
        # Diese Nebenbedingung gilt für mehrere Abholorte und mehrere Depots.
        model += (lpSum(x[(i, j, k)] for i in K for j in J for k in K) >= max(Q[i][j] for i in I for j in I),
                  f'11_Verlassen_des_Depots')

        ################################################################################################################
        # Berücksichtigung von Zeiten

        # Ankunftszeit am Depot ist immer 0
        for k in K:
            model += arrival_time[(k, k, k)] == 0, f"12_Ankunftszeit_Im_Depot_{k}_ist_0.0"

        # Ankunftszeit am ersten Knoten nach Verlassen des Depots
        for k in K:
            for j in J:
                # Ankunftszeit-Bedingung nur, wenn das Fahrzeug tatsächlich von i nach j fährt
                model += (arrival_time[(k, j, k)] == D[k][j] * x[(k, j, k)] / v,
                          (f"13_Ankunftszeit_Fahrzeug_{k}_von_Depot_{k}_zu_erstem_Knoten_{j}"))

        # Die Ankunftszeit eines Fahrzeugs an einem Knoten hängt von der Ankunftszeit am vorherigen Knoten, der Fahrzeit
        # zwischen den Knoten und der Wartezeit am vorherigen Knoten ab.
        for i in J:
            for j in J:
                for h in I:
                    if i != j and h != i and h != j:
                        for k in K:
                            if i != k:
                                # Ankunftszeit-Bedingung nur, wenn das Fahrzeug tatsächlich von i nach j fährt
                                model += arrival_time[(i, j, k)] >= arrival_time[(h, i, k)] + wait_time[(i, j, k)] + \
                                         D[i][j] * x[
                                             (i, j, k)] / v - M * (1 - x[(i, j, k)]), (
                                    f"14_Ankunftszeit_Fahrzeug_{k}_von_Knoten_"
                                    f"{i}_zu_Knoten_{j}_ausgehend_von_Knoten_{h}")

        # Die Ankunftszeit des Fahrzeugs, welches am längsten benötigt, ist entscheidend für die Transportdauer
        for i in J:
            for j in J:
                model += max_total_arrival_time >= max_arrival_time[(i, j)], f"16a_Transportdauer_Beschraenkung_{i}_{j}"
                for k in K:
                    # Maximale Ankunftszeit nur für Fahrzeuge, die tatsächlich zu Knoten j fahren
                    model += max_arrival_time[(i, j)] >= arrival_time[
                        (i, j, k)], f"15_Max_Ankunftszeit_Fahrzeug_{k}_Knoten_{j}_ausgehend_von_Knoten_{i}"
                    # model += max_total_arrival_time >= arrival_time[
                    #     (i, j, k)], f"16_Max_Ankunftszeit_Beschraenkung_{i}_{j}_{k}"

        for j in J:
            for i in J:
                for h in I:
                    if i != j and h != i and h != j:
                        for k in K:
                            # Wartezeit wird nur berechnet, wenn das Fahrzeug tatsächlich zu Knoten j fährt
                            model += (
                                wait_time[(i, j, k)] >= max_arrival_time[(i, j)] - arrival_time[(h, i, k)] - D[i][j] / v
                                - M * (1 - x[(h, i, k)]),
                                f"17_Wartezeit_Fahrzeug_{k}_von_Knoten_{i}_nach_Knoten_{j}_ausgehend_von_Knoten_{h}")
        #
        # # Die wait_time wird nur gesetzt, wenn ein Fahrzeug von Knoten i zu Knoten j fährt und ist sonst 0
        # for i in J:
        #     for j in J:
        #         if i != j:
        #             for k in K:
        #                 model += wait_time[(i, j, k)] <= M * x[(i, j, k)], \
        #                     f"Beschränkung_wait_time_nur_wenn_Fahrzeug_{k}_von_{i}_nach_{j}_fährt"
        #                 model += arrival_time[(i, j, k)] <= M * x[(i, j, k)], \
        #                     f"Beschränkung_arrival_time_nur_wenn_Fahrzeug_{k}_von_{i}_nach_{j}_fährt"
        #
        # # Die max_arrival_time wird nur gesetzt, wenn ein Fahrzeug von Knoten i zu Knoten j fährt und ist sonst 0
        # for i in J:
        #     for j in J:
        #         if i != j:
        #             model += max_arrival_time[(i, j)] <= M * lpSum(x[(i, j, k)] for k in K), \
        #                 f"Beschränkung_max_arrival_time_nur_wenn_Fahrzeug_von_{i}_nach_{j}_fährt"
        #
        # # Fahrzeuge dürfen nicht von Abholknoten zu Abholknoten fahren
        # for i in QQ:
        #     for j in QQ:
        #         model += max_arrival_time[
        #                      (i, j)] == 0, f"Verbot_von_Abholknoten_{i}_zu_Abholknoten_{j}_keine_Wartezeit"
        #         for k in range(A):
        #             model += wait_time[(
        #                 i, j,
        #                 k)] == 0, f"Verbot_Fahrzeug_{k}_von_Abholknoten_{i}_zu_Abholknoten_{j}_keine_Wartezeit"
        #             model += arrival_time[(
        #                 i, j,
        #                 k)] == 0, f"Verbot_Fahrzeug_{k}_von_Abholknoten_{i}_zu_Abholknoten_{j}_keine_Ankunftszeit"
        #
        # # Fahrzeuge dürfen nicht von Lieferknoten zu Lieferknoten fahren
        # for i in QL:
        #     for j in QL:
        #         model += max_arrival_time[
        #                      (i, j)] == 0, f"Verbot_von_Abholknoten_{i}_zu_Abholknoten_{j}_keine_Wartezeit"
        #         for k in range(A):
        #             model += wait_time[(
        #                 i, j,
        #                 k)] == 0, f"Verbot_Fahrzeug_{k}_von_Abholknoten_{i}_zu_Abholknoten_{j}_keine_Wartezeit"
        #             model += arrival_time[(
        #                 i, j,
        #                 k)] == 0, f"Verbot_Fahrzeug_{k}_von_Abholknoten_{i}_zu_Abholknoten_{j}_keine_Ankunftszeit"

        # warm_start_routes = self.warm_start(num_AGVs, Q)
        # for vehicle_path in warm_start_routes:
        #     x[(vehicle_path[0], vehicle_path[1], vehicle_path[2])].setInitialValue(1)

        # model.writeLP('model.lp')  # Ausgabe des Modells als Datei

        # Problem lösen
        # solver = CPLEX_CMD()
        # solver = GUROBI_CMD()
        # solver = GLPK_CMD(timeLimit=900)
        # solver = GLPK_CMD(options=["--mipgap 0.1", "--tmlim 10800"])
        # solver = GLPK_CMD(options=["--mipgap 0.01"])
        # solver = GLPK_CMD(options=["--tmlim 3600"])
        # solver = PULP_CBC_CMD(timeLimit=10800, gapRel=0.1)
        # solver = PULP_CBC_CMD(gapRel=0.1)
        # solver = CPLEX_CMD(gapRel=0.05)
        # solver = CPLEX_PY(gapRel=0.05)
        # solver = CPLEX_PY(gapRel=0.05, threads=1)
        # model.to_json('model.json')
        print(datetime.datetime.now())
        print('SOLVING...')
        print()
        solver = CPLEX_PY(warmStart=False)
        """
        WICHTIG: DIE PULP API WURDE BEARBEITET!!!
        C:\Program Files\Python310\Lib\site-packages\pulp\apis\cplex_api.py
        MIP GAP = 0.05
        NODEFILES ON DISK
        """
        # solver.buildSolverModel(model)
        # solver.solverModel.parameters.workdir.set("/tmp/cplex")
        # solver.solverModel.parameters.workmem.set(2048)
        # solver.solverModel.parameters.mip.strategy.file.set(3)
        # print(solver.solverModel.parameters.mip.strategy.file.get())
        # print(solver.solverModel.parameters.workdir.get())
        # solver = CPLEX_CMD(gapRel=0.05, options=["set timeLimit 9999999", "set mip strategy file 3", "set mip tolerances mipgap 0.05"])

        status = model.solve(solver)

        result = {
            'status': LpStatus[model.status],
            'objective': value(model.objective),
            'variables': {v.name: v.varValue for v in model.variables()}
        }

        # Printausgabe - Entscheidungsvariablen, wenn > 0
        # for v in model.variables():
        #     if v.name.startswith("arrival"):
        #         print(v.name, "=", v.varValue)
        #     if v.name.startswith("max"):
        #         print(v.name, "=", v.varValue)
        #     if v.name.startswith("wait"):
        #         print(v.name, "=", v.varValue)

        print("######")
        print('BEFORE ROUNDING')
        for v in model.variables():
            if v.varValue is not None:
                if v.varValue > 0:
                    print(v.name, "=", v.varValue)

        print("\nAFTER ROUNDING")
        for v in model.variables():
            if v.varValue is not None:
                if v.varValue > 0:
                    v.varValue = round(v.varValue, 1)
                    print(v.name, "=", v.varValue)

        # Printausgabe - Binärvariable x[(i, j, k)], wenn = 1
        '''for i in I:
            for j in I:
                for k in range(A):
                    if value(x[(i, j, k)]) == 1:
                        print(f"x({i}, {j}, {k}) = 1.0")'''

        # print(
        #     '###########################################################################################################')
        #
        # # Printausgabe, welches Fahrzeug von Knoten i zu Knoten k fährt
        # for k in range(A):
        #     for i in I:
        #         for j in I:
        #             if value(x[(i, j, k)]) == 1:
        #                 print(f"Fahrzeug {k} fährt von {i} nach {j} mit x({i}, {j}, {k}) = {value(x[(i, j, k)])}.")

        print(
            '###########################################################################################################')

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

        print(self.dict_result_1)
        print(self.x_values)
        # print(dict_result_2)
        # print(dict_result_2.items())

        # with open("data/Current_Factory/VRP/result_1.txt", "w") as convert_file:
        #     convert_file.write(json.dumps(self.dict_result_1))
        # with open("data/Current_Factory/VRP/result_1.json", "w") as outfile:
        #     json.dump(self.dict_result_1, outfile)
        #
        # with open("data/Current_Factory/VRP/result_2.txt", "w") as convert_file:
        #     convert_file.write(json.dumps(self.dict_result_2))
        # with open("data/Current_Factory/VRP/result_2.json", "w") as outfile:
        #     json.dump(self.dict_result_2, outfile)
        #
        # with open("data/Current_Factory/VRP/x_values.pkl", "wb") as outfile:
        #     pickle.dump(self.x_values, outfile)
        # with open("data/Current_Factory/VRP/x_values_2.pkl", "wb") as outfile:
        #     pickle.dump(self.x_values_2, outfile)

        self.determine_last_node()

        print(
            '###########################################################################################################')

        self.fitness = value(model.objective)
        print(f"Zielwert =\n{self.fitness}")
        print(f"Status:\n{LpStatus[model.status]}")
        try:
            self.mip_gap = solver.solverModel.solution.MIP.get_mip_relative_gap()
            print(f"MIP GAP =\n{self.mip_gap}")
        except:
            self.mip_gap = 0
            print(f"NO SOLUTION - NO MIP GAP")

        endtime = time()
        dauer = endtime - starttime
        print(f"Dauer der Optimierung = \n{dauer}")
        self.duration = dauer

        return self.x_values, self.x_values_2

    def warm_start(self, num_AGVs, Q):

        def complete_routes(lst):
            # Neue Liste zur Speicherung der vollständigen Route
            completed_list = []

            # Füge das erste Element hinzu
            completed_list.append(lst[0])

            print(f"JOBS: {lst}")

            # Durchlaufe die Liste und ergänze fehlende Verbindungen
            for i in range(len(lst) - 1):
                pickup = lst[i][1]  # Lieferort des vorherigen Eintrags
                delivery = lst[i + 1][0]  # Abholort des nächsten Eintrags
                vehicles = lst[i + 1][2]  # Anzahl an Fahrzeugen aus dem nächsten Eintrag

                # Füge den fehlenden Eintrag hinzu
                completed_list.append([pickup, delivery, vehicles])

                # Füge den nächsten Originaleintrag hinzu
                completed_list.append(lst[i + 1])

            return completed_list

        K = []  # Anzahl der Depots, wo die einzelnen FTF stehen - jedes FTF hat ein Depot
        I = []  # Orte/Knoten, die besucht werden (inkl. Depots, Abhol- und Lieferpunkten), wird aus der Distanzmatrix berechnet
        J = []  # Orte/Knoten der Jobs (inkl. Abhol- und Lieferpunkten)
        A = num_AGVs  # Anzahl der zur Verfügung stehenden Fahrzeuge
        v = 1

        for i in range(A):
            K.append(i)
        for i in range(len(Q)):
            I.append(i)
        num_locations = len(Q)
        J = list(set(I).difference(K))

        jobs = []

        for i in J:
            for j in J:
                if Q[i][j] > 0:
                    jobs.append([i, j, int(Q[i][j])])

        sorted_jobs = sorted(jobs, key=lambda x: x[-1], reverse=True)
        complete_sorted_jobs = complete_routes(sorted_jobs)

        warm_start = []

        for k in range(complete_sorted_jobs[0][2]):
            warm_start.append([k, complete_sorted_jobs[0][0], k])
        for i in range((len(complete_sorted_jobs))):
            for j in range(complete_sorted_jobs[i][2]):
                warm_start.append([complete_sorted_jobs[i][0], complete_sorted_jobs[i][1], j])

        print(warm_start)

        return warm_start

    def determine_last_node(self):
        # Ermittlung der letzten Knoten der Fahrzeuge
        end_nodes_per_vehicle = {}
        # Identifiziere alle vorhandenen Fahrzeug-IDs
        vehicle_ids = set(vehicle_id for _, _, vehicle_id in self.x_values.keys())
        for k in vehicle_ids:
            visited_edges = [(i, j) for i, j, vehicle_id in self.x_values.keys() if
                             self.x_values[(i, j, vehicle_id)] == 1.0 and vehicle_id == k]
            incoming_edges = {}
            outgoing_edges = {}
            for i, j in visited_edges:
                if j not in incoming_edges:
                    incoming_edges[j] = 0
                incoming_edges[j] += 1
                if i not in outgoing_edges:
                    outgoing_edges[i] = 0
                outgoing_edges[i] += 1
            end_nodes = [node for node in incoming_edges if
                         node not in outgoing_edges or incoming_edges[node] > outgoing_edges.get(node, 0)]
            end_nodes_per_vehicle[k] = end_nodes
        # Printausgabe, welches der letzten Knoten für jedes Fahrzeug ist
        print(f"Endknoten: {end_nodes_per_vehicle}")

    def reconstruct_path(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width * self.pixel_size, self.height * self.pixel_size))
        pygame.display.set_caption('Path for every AGV')
        color_grid = self.factory.get_color_grid()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
            self.display_colors(color_grid)
            self.display_nodes()
            self.display_factory_objects_name()
            # self.draw_paths()
            self.draw_grid()
            self.wait()

    def wait(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return

    def display_colors(self, color_data):
        # Draw the color data
        for y in range(self.height):
            for x in range(self.width):
                pygame.draw.rect(self.screen, color_data[x][y],
                                 (x * self.pixel_size, y * self.pixel_size, self.pixel_size, self.pixel_size))
                # if color_data[x][y] != [255, 255, 5]:
                #     pygame.draw.rect(self.screen, (0, 0, 0),
                #                      (x * self.pixel_size, y * self.pixel_size, self.pixel_size, self.pixel_size), 1)

        # Draw the AGVs
        for agv in self.factory.agvs:
            agv_pos = [agv.pos_x, agv.pos_y]
            pygame.draw.rect(self.screen, [0, 0, 0],
                             (6 + agv_pos[0] * self.pixel_size + (agv.width / self.reference_size * self.pixel_size) / 2,
                                   agv_pos[1] * self.pixel_size + (agv.width / self.reference_size * self.pixel_size) / 2,
                                   agv.width / self.reference_size * self.pixel_size,
                                   agv.length / self.reference_size * self.pixel_size),
                             border_radius=2)

        pygame.display.flip()

    def draw_grid(self):
        for y in range(self.width):
            pygame.draw.line(self.screen, [0, 0, 0], [0, y * self.pixel_size + self.pixel_size],
                             [self.width * self.pixel_size, y * self.pixel_size + self.pixel_size])
        for x in range(self.width):
            pygame.draw.line(self.screen, [0, 0, 0], [x * self.pixel_size + self.pixel_size, 0],
                             [x * self.pixel_size + self.pixel_size, self.height * self.pixel_size])
        pygame.display.flip()

    def display_nodes(self):
        node_position = self.assignment_node_to_agv_warehouse_machine()
        for key, value in node_position.items():
            font = pygame.font.SysFont('arial', 12)
            text = font.render(str(key), True, (0, 0, 0))
            self.screen.blit(text, [self.pixel_size * value[0] + 2, self.pixel_size * value[1] + 2])

        pygame.display.flip()

    def display_factory_objects_name(self):
        # Draw the names of the factory objects
        for machine in self.factory.machines:
            font = pygame.font.SysFont('arial', 12)
            text = font.render(machine.name, True, (200, 200, 200))
            # print(text)
            # print(self.pixel_size * machine.pos_x + 2, self.pixel_size * machine.pos_y)
            self.screen.blit(text, [self.pixel_size * machine.pos_x + 2, self.pixel_size * machine.pos_y + 2])

        for warehouse in self.factory.warehouses:
            font = pygame.font.SysFont('arial', 12)
            text = font.render(warehouse.name, True, (200, 200, 200))
            self.screen.blit(text, [self.pixel_size * warehouse.pos_x + 2, self.pixel_size * warehouse.pos_y + 2])

        pygame.display.flip()

    def assignment_node_to_agv_warehouse_machine(self):
        dict_assignment_factory_object = dict()
        dict_assignment_node_position = dict()

        for i in range(self.dimension):
            if isinstance(self.list_agv_input_output[i][2], AGV):
                dict_assignment_factory_object[i] = self.list_agv_input_output[i][2]
                dict_assignment_node_position[i] = [self.list_agv_input_output[i][2].pos_x,
                                                    self.list_agv_input_output[i][2].pos_y]
            else:
                if self.list_agv_input_output[i][0] == 'input':
                    dict_assignment_factory_object[i] = self.list_agv_input_output[i][2]
                    dict_assignment_node_position[i] = self.list_agv_input_output[i][2].pos_input
                if self.list_agv_input_output[i][0] == 'output':
                    dict_assignment_factory_object[i] = self.list_agv_input_output[i][2]
                    dict_assignment_node_position[i] = self.list_agv_input_output[i][2].pos_output

        # for i in range(self.dimension):
        #     if isinstance(self.list_of_factory_objects_input_output[i], AGV):
        #         dict_assignment_factory_object[i] = self.list_of_factory_objects_input_output[i]
        #         dict_assignment_node_position[i] = [self.list_of_factory_objects_input_output[i].pos_x,
        #                                             self.list_of_factory_objects_input_output[i].pos_y]
        #     else:
        #         if self.list_of_factory_objects_input_output[i] != self.list_of_factory_objects_input_output[i - 1]:
        #             dict_assignment_factory_object[i] = self.list_of_factory_objects_input_output[i]
        #             dict_assignment_node_position[i] = self.list_of_factory_objects_input_output[i].pos_input
        #         else:
        #             dict_assignment_factory_object[i] = self.list_of_factory_objects_input_output[i]
        #             dict_assignment_node_position[i] = self.list_of_factory_objects_input_output[i].pos_output
        return dict_assignment_node_position

    def draw_paths(self):
        node_position = self.assignment_node_to_agv_warehouse_machine()
        #print(node_position)
        #print(self.x_values)
        number_of_colors = len(self.factory.agvs)
        color = [list(np.random.choice(range(256), size=3)) for _ in range(number_of_colors)]
        #print(color)
        vehicle_ids = set(vehicle_id for _, _, vehicle_id in self.x_values.keys())
        visited_edges = []
        #print(vehicle_ids)
        for k in vehicle_ids:
            visited_edges.append(self.get_route_for_vehicle(self.dict_order_of_transport, k))
            #visited_edges.append([(i, j) for i, j, vehicle_id in self.x_values.keys() if
            #                      self.x_values[(i, j, vehicle_id)] == 1.0 and vehicle_id == k])
            print(f'Fahrzeug k = {k}, visited_edges = {visited_edges[k]}')
            i = 0
            step = 0
            for line in visited_edges[k]:
                # print(f'Linie: {line}')
                # print(f'Knotenposition {line[0]} = {node_position[line[0]]}')
                # print(f'Knotenposition {line[1]} = {node_position[line[1]]}')
                # displays every agv:
                pygame.draw.line(self.screen, color[k],
                                 [node_position[line[0]][0] * self.pixel_size + self.pixel_size // 2 + 2 * k,
                                  node_position[line[0]][1] * self.pixel_size + self.pixel_size // 2 + 2 * k],
                                 [node_position[line[1]][0] * self.pixel_size + self.pixel_size // 2 + 2 * k,
                                  node_position[line[1]][1] * self.pixel_size + self.pixel_size // 2 + 2 * k], 2)
                # if k == 3:
                #     pygame.draw.line(self.screen, color[k],
                #                      [node_position[line[0]][0] * self.pixel_size + self.pixel_size // 4 + i - 4,
                #                       node_position[line[0]][1] * self.pixel_size + self.pixel_size // 4],
                #                      [node_position[line[1]][0] * self.pixel_size + self.pixel_size // 4 + i,
                #                       node_position[line[1]][1] * self.pixel_size + self.pixel_size // 4], 2)
                #     i += 6
                    # pygame.draw.line(self.screen, color[k],
                    #                  [node_position[line[0]][0] * self.pixel_size + self.pixel_size // 2,
                    #                   node_position[line[0]][1] * self.pixel_size + self.pixel_size // 2],
                    #                  [node_position[line[1]][0] * self.pixel_size + self.pixel_size // 2,
                    #                   node_position[line[1]][1] * self.pixel_size + self.pixel_size // 2], 2)
                # font = pygame.font.SysFont('arial', 12)
                #     text = font.render(str(step), True, (0, 0, 0))
                #     #self.screen.blit(text, [ , self.pixel_size * ])
                #     step += 1

        pygame.display.flip()


    def get_order_of_transport_new(self):
        dict_order = dict() # in the dict "dict_order" the order shall be saved as a dict,
        #                      the step number is the key and the routes (saved as (i,j,k)) are the values
        dict_order, remaining_x_values, actual_node_position = self.get_first_step_of_transport()
        step = 1
        # print('#######################################################################################################')
        # print('#######################################################################################################')
        # print('#######################################################################################################')
        # print(f'Verbleibende Kanten gesamt: {remaining_x_values}')
        # print(f'Aktuelle Reihenfolge des Transports: {dict_order}')
        # print(f'Positionen der Fahrzeuge nach Iteration 0: {actual_node_position}')
        # print('#######################################################################################################')
        remaining_x_values_dict_per_step = dict()
        remaining_x_values_dict_per_step[step] = remaining_x_values
        actual_node_position_dict_per_step = dict()
        actual_node_position_dict_per_step[step] = actual_node_position

        if self.get_next_step_of_transport(remaining_x_values_dict_per_step, dict_order, actual_node_position_dict_per_step, step)  == False:
            print('There is no solution\n')
            return False
        else:
            self.dict_order_of_transport = dict_order
            print(f'Solution: {dict_order}')
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
                actual_node_position[k] = j  # sets the current node position fpr each vehicle k
            dict_order[0] = working_list_order  # adds the travelled edge to dict

        return dict_order, remaining_x_values, actual_node_position

    def get_next_step_of_transport(self, remaining_x_values_dict_per_step, dict_order, actual_node_position_dict_per_step, step):
        # print(remaining_x_values_dict_per_step)
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
            #print('##########')

        # print(f'Mögliche Kanten: {possible_vehicle_edges}')

        num_vehicles = 0
        necessary_vehicles_for_transport = self.count_tuple_combinations(remaining_x_values_dict_per_step[step])
        # print(f'Necessary vehicles for transport: {necessary_vehicles_for_transport}')
        possible_edges = []
        for i in range(self.dimension):  # jeder Knoten i wird durchgegangen
            for k in actual_node_position_dict_per_step[step].values():
                if k == i:
                    num_vehicles += 1  # wird +1, wenn ein Fahrzeug sich am Knoten i befindet
            for key, value in necessary_vehicles_for_transport.items():
                if value <= num_vehicles and i == key[0]:
                    # print(f'Knoten = {i}, Fahrzeuge = {num_vehicles}')
                    # print(f'Mögliche Fahrt: {key}')
                    possible_edges.append(key)
            num_vehicles = 0
        # print(f'MÖGLICHE Fahrten gesamt: {possible_edges}')

        for possible_edge in possible_edges:
            working_list_order = []  # temporary list, which contains each x(i,j,k) for each step of the order as tuples
            # print(
            #     '\n#######################################################################################################\n')
            # print(f'Iteration {step}')
            # print(f'Mögliche Kante: {possible_edge}')
            # print(f'Verbleibende Kanten gesamt: {remaining_x_values_dict_per_step[step]}')
            # print(f'Positionen der Fahrzeuge vor Iteration {step}: {actual_node_position_dict_per_step[step]}')
            remaining_x_values_dict_per_step[step + 1] = remaining_x_values_dict_per_step[step].copy()
            actual_node_position_dict_per_step[step + 1 ] = actual_node_position_dict_per_step[step].copy()
            for i, j, k in remaining_x_values_dict_per_step[step]:
                if i == possible_edge[0] and j == possible_edge[1]:
                    working_list_order.append((i, j, k))  # adds the paths as a tuple
                    remaining_x_values_dict_per_step[step+1].pop((i, j, k))
                    actual_node_position_dict_per_step[step + 1][k] = j  # sets the current node position for each vehicle k
            dict_order[step]  = working_list_order
            len_dict_order = len(dict_order)
            # print(f'{step}, {len_dict_order}')
            if len(dict_order) > step:
                for i in range(len_dict_order, step):
                    dict_order.pop(i)
            # print(f'Ausgewählte Kante: {possible_edge}')
            # print(f'Verbleibende Kanten gesamt nach Fahrt: {remaining_x_values_dict_per_step[step + 1]}')
            # print(f'Aktuelle Reihenfolge des Transports: {dict_order}')
            # print(f'Positionen der Fahrzeuge nach Iteration {step}: {actual_node_position_dict_per_step[step + 1]}')
            # print(
            #     '\n#######################################################################################################\n')
            if self.get_next_step_of_transport(remaining_x_values_dict_per_step, dict_order, actual_node_position_dict_per_step, step + 1) == True:
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


    def get_dynamic_routing(self):
        self.create_file_for_list_of_factory_objects()
        self.create_dataframe_of_factory_objects()
        D = self.get_distance_matrix()
        # vrp.get_delivery_relationship()
        # Q = vrp.get_amount_of_agv_for_delivery_as_matrix_free_configuration()  # Konfiguration der FTF
        Q = self.get_amount_of_agv_for_delivery_as_matrix_1_4_6_configuration()  # Konfiguration der FTF

        self.cVRPPDmDnR(num_AGVs=len(self.factory.agvs), D=D, Q=Q)

        return self.get_order_of_transport_new()




########################################################################################################################
###   BEISPIEL ALEX
########################################################################################################################

def create_default_factory(self):
    '''
    This is the default factory created with Alex.
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

    self.warehouses.append(Warehouse())
    self.warehouses[0].name = "Warehouse_0"
    self.warehouses[0].pos_x = 0
    self.warehouses[0].pos_y = 8
    self.warehouses[0].length = 5
    self.warehouses[0].width = 2
    self.warehouses[0].pos_input = [4, 8]
    self.warehouses[0].pos_output = [1, 8]
    self.warehouses[0].input_products = ['four']  # ['four']
    self.warehouses[0].output_products = ['one']
    self.warehouses[0].factory = self
    self.warehouses[0].process_time = 10
    self.warehouses[0].rest_process_time = 10

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
    self.machines[0].factory = self

    self.machines.append(Machine())
    self.machines[1].name = "Machine_1"
    self.machines[1].pos_x = 7
    self.machines[1].pos_y = 0
    self.machines[1].length = 3
    self.machines[1].width = 3
    self.machines[1].pos_input = [7, 1]
    self.machines[1].pos_output = [8, 2]
    self.machines[1].input_products = ['two']
    self.machines[1].output_products = ['three']
    self.machines[1].factory = self
    self.machines[1].process_time = 20
    self.machines[1].rest_process_time = 20

    self.machines.append(Machine())
    self.machines[2].name = "Machine_2"
    self.machines[2].pos_x = 7
    self.machines[2].pos_y = 7
    self.machines[2].length = 3
    self.machines[2].width = 3
    self.machines[2].pos_input = [8, 7]
    self.machines[2].pos_output = [7, 8]
    self.machines[2].input_products = ['three']
    self.machines[2].output_products = ['four']
    self.machines[2].factory = self
    self.machines[2].process_time = 10
    self.machines[2].rest_process_time = 10

    self.loading_stations.append(LoadingStation())
    self.loading_stations[0].pos_x = 0
    self.loading_stations[0].pos_y = 7
    self.loading_stations[0].length = 1
    self.loading_stations[0].width = 1
    self.agvs.append(AGV([0, 7]))
    self.agvs[0].name = 0
    self.agvs[0].thread_running = False
    self.agvs[0].factory = self

    self.loading_stations.append(LoadingStation())
    self.loading_stations[1].pos_x = 0
    self.loading_stations[1].pos_y = 6
    self.loading_stations[1].length = 1
    self.loading_stations[1].width = 1
    self.agvs.append(AGV([0, 6]))
    self.agvs[1].name = 1
    self.agvs[1].thread_running = False
    self.agvs[1].factory = self

    self.loading_stations.append(LoadingStation())
    self.loading_stations[2].pos_x = 0
    self.loading_stations[2].pos_y = 5
    self.loading_stations[2].length = 1
    self.loading_stations[2].width = 1
    self.agvs.append(AGV([0, 5]))
    self.agvs[2].name = 2
    self.agvs[2].thread_running = False
    self.agvs[2].factory = self

    self.loading_stations.append(LoadingStation())
    self.loading_stations[3].pos_x = 0
    self.loading_stations[3].pos_y = 4
    self.loading_stations[3].length = 1
    self.loading_stations[3].width = 1
    self.agvs.append(AGV([0, 4]))
    self.agvs[3].name = 3
    self.agvs[3].thread_running = False
    self.agvs[3].factory = self

    self.loading_stations.append(LoadingStation())
    self.loading_stations[4].pos_x = 5
    self.loading_stations[4].pos_y = 9
    self.loading_stations[4].length = 1
    self.loading_stations[4].width = 1
    self.agvs.append(AGV([5, 9]))
    self.agvs[4].name = 4
    self.agvs[4].thread_running = False
    self.agvs[4].factory = self

    self.loading_stations.append(LoadingStation())
    self.loading_stations[5].pos_x = 6
    self.loading_stations[5].pos_y = 9
    self.loading_stations[5].length = 1
    self.loading_stations[5].width = 1
    self.agvs.append(AGV([6, 9]))
    self.agvs[5].name = 5
    self.agvs[5].thread_running = False
    self.agvs[5].factory = self

    self.fill_grid()
    print(self.factory_grid_layout)

    self.product_types['one'] = dict(length=1100, width=600, weight=4.5)  # dict(length=1100, width=600, weight=9.0)
    self.product_types['two'] = dict(length=600, width=600, weight=4.5)  # dict(length=600, width=600, weight=4.5)
    self.product_types['three'] = dict(length=250, width=250, weight=4.5)
    self.product_types['four'] = dict(length=250, width=250, weight=4.5)
    print(self.product_types)
    print(self.machines)


########################################################################################################################
###   GITTERROST PRODUKTION
########################################################################################################################
def create_default_factory_2(self):
    '''
    This is a factory to produce Gitterroste
    :param self:
    :return:
    '''
    print('############################################')
    print('##### Creating default factory for VRP #####')
    print('############################################')
    self.length = 200
    self.width = 100
    self.cell_size = 10
    self.no_columns = int(self.length // self.cell_size)
    self.no_rows = int(self.width // self.cell_size)
    self.factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows)).tolist()

    # Produkte innerhalb der Fabrik
    self.product_types['extern_0'] = dict(length=0.1, width=0.1, weight=0.1)  # in: W0
    self.product_types['extern_WA'] = dict(length=0.1, width=0.1, weight=0.1)  # out: W7
    self.product_types['coil'] = dict(length=1.5, width=0.8, weight=1000)  # out: W0 | in: M0, M1, M2
    self.product_types['anbauteile'] = dict(length=1.2, width=0.8, weight=200)  # out: W0 | in: M15
    self.product_types['randeinfassungen_1'] = dict(length=1, width=0.8, weight=50)  # out: W0 | in: W6
    self.product_types['spaltband_1'] = dict(length=1, width=0.8, weight=100)  # out: M0, M1, M2 | in: W1
    self.product_types['spaltband_2'] = dict(length=1, width=0.8, weight=100)  # out: W1 | in: M3, M4, M5, M6
    self.product_types['gestanzt_spaltband_1'] = dict(length=1, width=0.8, weight=95)  # out: M3, M4 | in: W2
    self.product_types['gestanzt_spaltband_2'] = dict(length=1, width=0.8, weight=95)  # out: W2 | in: M5, M6
    self.product_types['abfall'] = dict(length=0.5, width=0.4, weight=20)  # out: M3, M4, M13 | in: W5
    self.product_types['extern_1'] = dict(length=0.1, width=0.1, weight=0.1)  # out: W5
    self.product_types['abgelaengtes_spaltband_1'] = dict(length=1, width=0.5, weight=50)  # out: M5, M6 | in: W3
    self.product_types['abgelaengtes_spaltband_2'] = dict(length=1, width=0.5, weight=50)  # out: W3 | in: M7, M8, M9
    # self.product_types['rohrost'] = dict(length=1, width=0.8, weigth=20)  # out: M7, M8, M9 | in: M10, M11, M12
    self.product_types['rohrost_1'] = dict(length=1, width=0.8, weigth=20)  # out: M7 | in: M10
    self.product_types['rohrost_2'] = dict(length=1, width=0.8, weigth=20)  # out: M8 | in: M11
    self.product_types['rohrost_3'] = dict(length=1, width=0.8, weigth=20)  # out: M9 | in: M12
    self.product_types['randeinfassungen_2'] = dict(length=1, width=0.8, weight=50)  # out: W6 | in: M10, M11, M12
    self.product_types['rost'] = dict(length=1, width=0.8, weight=20)  # out: M10, M11, M12 | in: W4
    self.product_types['fertiger_rost'] = dict(length=1, width=0.8, weight=20)  # out: W4 | in: W7
    self.product_types['rost_weiterbearbeitung'] = dict(length=1, width=0.8, weight=20)  # out: W4 | in: M13, M14
    self.product_types['rost_geschnitten'] = dict(length=1, width=0.8, weight=20)  # out: M13 | in: M14

    # LAGER / WARENHÄUSER

    # Wareneingangslager
    self.warehouses.append(Warehouse())
    self.warehouses[0].name = 'Wareneingang'
    self.warehouses[0].pos_x = 0
    self.warehouses[0].pos_y = 0
    self.warehouses[0].input_products = ['extern_0']
    self.warehouses[0].output_products = ['coil', 'anbauteile', 'randeinfassungen_1']
    self.warehouses[0].factory = self

    # Zwischenlager 1 - Spaltbänder
    self.warehouses.append(Warehouse())
    self.warehouses[1].name = 'Zwischenlager - Spaltbänder'
    self.warehouses[1].pos_x = 50
    self.warehouses[1].pos_y = 50
    self.warehouses[1].input_products = ['spaltband_1']
    self.warehouses[1].output_products = ['spaltband_2']
    self.warehouses[1].factory = self

    # Zwischenlager 2 - gestanzte Spaltbänder
    self.warehouses.append(Warehouse())
    self.warehouses[2].name = 'Zwischenlager - gestanzte Spaltbänder'
    self.warehouses[2].pos_x = 80
    self.warehouses[2].pos_y = 90
    self.warehouses[2].input_products = ['gestanzt_spaltband_1']
    self.warehouses[2].output_products = ['gestanzt_spaltband_2']
    self.warehouses[2].factory = self

    # Zwischenlager 3 - abgelängte Spaltbänder
    self.warehouses.append(Warehouse())
    self.warehouses[3].name = 'Zwischenlager - abgelängte Spaltbänder'
    self.warehouses[3].pos_x = 100
    self.warehouses[3].pos_y = 10
    self.warehouses[3].input_products = ['abgelaengtes_spaltband_1']
    self.warehouses[3].output_products = ['abgelaengtes_spaltband_2']
    self.warehouses[3].factory = self

    # Zwischenlager 4 - Roste
    self.warehouses.append(Warehouse())
    self.warehouses[4].name = 'Zwischenlager - Roste'
    self.warehouses[4].pos_x = 160
    self.warehouses[4].pos_y = 50
    self.warehouses[4].input_products = ['rost']
    self.warehouses[4].output_products = ['fertiger_rost', 'rost_weiterbearbeitung']
    self.warehouses[4].factory = self

    # Zwischenlager 5 - Abfall gestanzte Spaltbänder
    self.warehouses.append(Warehouse())
    self.warehouses[5].name = 'Metallschrott'
    self.warehouses[5].pos_x = 100
    self.warehouses[5].pos_y = 100
    self.warehouses[5].input_products = ['abfall']
    self.warehouses[5].output_products = ['extern_1']
    self.warehouses[5].factory = self

    # Zwischenlager 6 - Randeinfassungen
    self.warehouses.append(Warehouse())
    self.warehouses[6].name = 'Zwischenlager - Randeinfassung'
    self.warehouses[6].pos_x = 140
    self.warehouses[6].pos_y = 0
    self.warehouses[6].input_products = ['randeinfassungen_1']
    self.warehouses[6].output_products = ['randeinfassungen_2']
    self.warehouses[6].factory = self

    # Warenausgangslager (7)
    self.warehouses.append(Warehouse())
    self.warehouses[7].name = 'Warenausgang'
    self.warehouses[7].pos_x = 200
    self.warehouses[7].pos_y = 0
    self.warehouses[7].input_products = ['fertiger_rost']
    self.warehouses[7].output_products = ['extern_WA']
    self.warehouses[7].factory = self

    # MASCHINEN

    # Maschine 0 - Coil spalten
    self.machines.append(Machine())
    self.machines[0].name = 'Spaltanlage 1'
    self.machines[0].pos_x = 20
    self.machines[0].pos_y = 20
    self.machines[0].input_products = ['coil']
    self.machines[0].output_products = ['spaltband_1']
    self.machines[0].factory = self

    # Maschine 1 - Coil spalten
    self.machines.append(Machine())
    self.machines[1].name = 'Spaltanlage 2'
    self.machines[1].pos_x = 20
    self.machines[1].pos_y = 50
    self.machines[1].input_products = ['coil']
    self.machines[1].output_products = ['spaltband_1']
    self.machines[1].factory = self

    # Maschine 2 - Coil spalten
    self.machines.append(Machine())
    self.machines[2].name = 'Spaltanlage 3'
    self.machines[2].pos_x = 20
    self.machines[2].pos_y = 80
    self.machines[2].input_products = ['coil']
    self.machines[2].output_products = ['spaltband_1']
    self.machines[2].factory = self

    # Maschine 3 - Stanzen
    self.machines.append(Machine())
    self.machines[3].name = 'Stanzanlage 1'
    self.machines[3].pos_x = 50
    self.machines[3].pos_y = 90
    self.machines[3].input_products = ['spaltband_2']
    self.machines[3].output_products = ['gestanzt_spaltband_1', 'abfall']
    self.machines[3].factory = self

    # Maschine 4 - Stanzen
    self.machines.append(Machine())
    self.machines[4].name = 'Stanzanlage 2'
    self.machines[4].pos_x = 70
    self.machines[4].pos_y = 90
    self.machines[4].input_products = ['spaltband_2']
    self.machines[4].output_products = ['gestanzt_spaltband_1', 'abfall']
    self.machines[4].factory = self

    # Maschine 5 - Ablängen
    self.machines.append(Machine())
    self.machines[5].name = 'Ablänganlage 1'
    self.machines[5].pos_x = 80
    self.machines[5].pos_y = 50
    self.machines[5].input_products = ['spaltband_2', 'gestanzt_spaltband_2']
    self.machines[5].output_products = ['abgelaengtes_spaltband_1']
    self.machines[5].factory = self

    # Maschine 6 - Ablängen
    self.machines.append(Machine())
    self.machines[6].name = 'Ablänganlage 2'
    self.machines[6].pos_x = 80
    self.machines[6].pos_y = 20
    self.machines[6].input_products = ['spaltband_2', 'gestanzt_spaltband_2']
    self.machines[6].output_products = ['abgelaengtes_spaltband_1']
    self.machines[6].factory = self

    # Maschine 7 - Setzen
    self.machines.append(Machine())
    self.machines[7].name = 'Setzanlage 1'
    self.machines[7].pos_x = 110
    self.machines[7].pos_y = 20
    self.machines[7].input_products = ['spaltband_2', 'abgelaengtes_spaltband_2']
    self.machines[7].output_products = ['rohrost_1']
    self.machines[7].factory = self

    # Maschine 8 - Setzen
    self.machines.append(Machine())
    self.machines[8].name = 'Setzanlage 2'
    self.machines[8].pos_x = 110
    self.machines[8].pos_y = 50
    self.machines[8].input_products = ['spaltband_2', 'abgelaengtes_spaltband_2']
    self.machines[8].output_products = ['rohrost_2']
    self.machines[8].factory = self

    # Maschine 9 - Setzen
    self.machines.append(Machine())
    self.machines[9].name = 'Setzanlage 3'
    self.machines[9].pos_x = 110
    self.machines[9].pos_y = 80
    self.machines[9].input_products = ['spaltband_2', 'abgelaengtes_spaltband_2']
    self.machines[9].output_products = ['rohrost_3']
    self.machines[9].factory = self

    # Maschine 10 - Randeinfassen
    self.machines.append(Machine())
    self.machines[10].name = 'Randeinfassung 1'
    self.machines[10].pos_x = 150
    self.machines[10].pos_y = 20
    self.machines[10].input_products = ['rohrost_1', 'randeinfassungen_2']
    self.machines[10].output_products = ['rost']
    self.machines[10].factory = self

    # Maschine 11 - Randeinfassen
    self.machines.append(Machine())
    self.machines[11].name = 'Randeinfassung 2'
    self.machines[11].pos_x = 150
    self.machines[11].pos_y = 50
    self.machines[11].input_products = ['rohrost_2', 'randeinfassungen_2']
    self.machines[11].output_products = ['rost']
    self.machines[11].factory = self

    # Maschine 12 - Randeinfassen
    self.machines.append(Machine())
    self.machines[12].name = 'Randeinfassung 3'
    self.machines[12].pos_x = 150
    self.machines[12].pos_y = 80
    self.machines[12].input_products = ['rohrost_3', 'randeinfassungen_2']
    self.machines[12].output_products = ['rost']
    self.machines[12].factory = self

    # Maschine 13 - Schneiden
    self.machines.append(Machine())
    self.machines[13].name = 'Schneidanlage 1'
    self.machines[13].pos_x = 160
    self.machines[13].pos_y = 90
    self.machines[13].input_products = ['rost_weiterbearbeitung']
    self.machines[13].output_products = ['rost_geschnitten', 'abfall']
    self.machines[13].factory = self

    # Maschine 14 - Sonder-Randeinfassung
    self.machines.append(Machine())
    self.machines[14].name = 'Sonder-Randeinfassung 1'
    self.machines[14].pos_x = 190
    self.machines[14].pos_y = 90
    self.machines[14].input_products = ['rost_geschnitten']
    self.machines[14].output_products = ['fertiger_rost']
    self.machines[14].factory = self

    # LOADING STATIONS - DEPOTS der AGVs

    # Version 1 - pro Ladestation 1 AGV
    i = 0
    n = 10  # Anzahl der Ladestationen = Anzahl der AGVs - pro Ladestation 1 AGV
    for i in range(n):
        self.loading_stations.append(LoadingStation)
        self.loading_stations[i].pos_x = 0
        self.loading_stations[i].pos_y = 10 * i + 20
        self.loading_stations[i].name = f'Loading_Station_{i}'
        self.agvs.append(AGV([0, 10 * i + 20]))
        self.agvs[i].thread_running = False
        self.agvs[i].name = i
        self.agvs[i].factory = self

    # Version 2 - eine Ladestation als Depot mit allen AGVs
    '''self.loading_stations.append(LoadingStation)
    self.loading_stations[0].name = 'Depot_0_AGVs'
    self.loading_stations[0].pos_x = 0
    self.loading_stations[0].pos_y = 20
    i = 0
    n = 10 # Anzahl der AGVs
    for i in range(n):
        self.agvs.append(AGV([0, 20]))
        self.agvs[i].factory = self'''


########################################################################################################################
###   BEISPIEL mit 3 Maschinen
########################################################################################################################

def create_default_factory_3(self):
    '''

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

    self.product_types['lager_m1'] = dict(length=1.5, width=1, weight=4.5)
    self.product_types['m1_m2'] = dict(length=1, width=1, weight=4.5)
    self.product_types['m2_m3'] = dict(length=0.5, width=0.5, weight=4.5)
    self.product_types['m3_lager'] = dict(length=0.5, width=0.5, weight=4.5)
    print(self.product_types)

    self.warehouses.append(Warehouse())
    self.warehouses[0].name = 'W1'
    self.warehouses[0].pos_x = 0
    self.warehouses[0].pos_y = 8
    self.warehouses[0].length = 5
    self.warehouses[0].width = 2
    self.warehouses[0].pos_input = [4, 8]
    self.warehouses[0].pos_output = [1, 8]
    self.warehouses[0].input_products = ['m3_lager']  # ['four']
    self.warehouses[0].output_products = ['lager_m1']
    self.warehouses[0].factory = self

    self.machines.append(Machine())
    self.machines[0].name = 'M1'
    self.machines[0].pos_x = 0
    self.machines[0].pos_y = 0
    self.machines[0].length = 3
    self.machines[0].width = 3
    self.machines[0].pos_input = [1, 2]
    self.machines[0].pos_output = [2, 1]
    self.machines[0].input_products = ['lager_m1']
    self.machines[0].output_products = ['m1_m2']
    self.machines[0].factory = self

    self.machines.append(Machine())
    self.machines[1].name = 'M2'
    self.machines[1].pos_x = 7
    self.machines[1].pos_y = 0
    self.machines[1].length = 3
    self.machines[1].width = 3
    self.machines[1].pos_input = [7, 1]
    self.machines[1].pos_output = [8, 2]
    self.machines[1].input_products = ['m1_m2']
    self.machines[1].output_products = ['m2_m3']
    self.machines[1].factory = self

    self.machines.append(Machine())
    self.machines[2].name = 'M3'
    self.machines[2].pos_x = 7
    self.machines[2].pos_y = 7
    self.machines[2].length = 3
    self.machines[2].width = 3
    self.machines[2].pos_input = [8, 7]
    self.machines[2].pos_output = [7, 8]
    self.machines[2].input_products = ['m2_m3']
    self.machines[2].output_products = ['m3_lager']
    self.machines[2].factory = self

    self.loading_stations.append(LoadingStation())
    self.loading_stations[0].pos_x = 0
    self.loading_stations[0].pos_y = 7
    self.loading_stations[0].length = 1
    self.loading_stations[0].width = 1

    i = 0
    n = 6  # Anzahl der AGVs
    for i in range(n):
        self.agvs.append(AGV([0, 7]))
        self.agvs[i].thread_running = False
        self.agvs[i].factory = self


def create_default_factory_4(self):
    '''
    This function creates a default factory, where 4 machines get deliveries from a warehouse
    ________
    |     1|
    |      |
    |      |
    |D W  2|
    |      |
    |      |
    |    3 |
    :param self:
    :return:
    '''
    print('############################################')
    print('##### Creating default factory for VRP #####')
    print('############################################')
    self.length = 8
    self.width = 10
    self.no_columns = int(self.length // self.cell_size)
    self.no_rows = int(self.width // self.cell_size)
    self.factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows)).tolist()

    self.product_types['lager_m1'] = dict(length=1500, width=1000, weight=1)
    # self.product_types['lager_m2'] = dict(length=1.5, width=0.5, weight=4.5)
    # self.product_types['lager_m3'] = dict(length=0.5, width=0.5, weight=4.5)
    # self.product_types['m1_m2'] = dict(length=1, width=1, weight=4.5)
    self.product_types['m2_m3'] = dict(length=1500, width=1000, weight=1)
    self.product_types['m3_m2'] = dict(length=1500, width=1000, weight=1)
    # self.product_types['m3_m4'] = dict(length=1500, width=1000, weight=1)
    # self.product_types['m4_m2'] = dict(length=1500, width=1000, weight=1)
    # self.product_types['m3_lager'] = dict(length=0.5, width=0.5, weight=4.5)
    print(self.product_types)

    self.warehouses.append(Warehouse())
    self.warehouses[0].pos_x = 2
    self.warehouses[0].pos_y = 3
    self.warehouses[0].length = 1
    self.warehouses[0].width = 2
    self.warehouses[0].pos_input = [2, 4]
    self.warehouses[0].pos_output = [2, 3]
    self.warehouses[0].input_products = ['empty_w1']
    self.warehouses[0].output_products = ['lager_m1']
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
    self.machines[0].output_products = ['emtpy_m1']
    self.machines[0].factory = self

    self.machines.append(Machine())
    self.machines[1].name = 'M2'
    self.machines[1].pos_x = 5
    self.machines[1].pos_y = 3
    self.machines[1].length = 1
    self.machines[1].width = 2
    self.machines[1].pos_input = [5, 4]
    self.machines[1].pos_output = [5, 3]
    self.machines[1].input_products = ['m3_m2']
    self.machines[1].output_products = ['m2_m3']
    self.machines[1].factory = self

    self.machines.append(Machine())
    self.machines[2].name = 'M3'
    self.machines[2].pos_x = 5
    self.machines[2].pos_y = 8
    self.machines[2].length = 1
    self.machines[2].width = 2
    self.machines[2].pos_input = [5, 9]
    self.machines[2].pos_output = [5, 8]
    self.machines[2].input_products = ['m2_m3']
    self.machines[2].output_products = ['m3_m2']
    self.machines[2].factory = self

    '''self.machines.append(Machine())
    self.machines[3].name = 'M4'
    self.machines[3].pos_x = 7
    self.machines[3].pos_y = 4
    self.machines[3].length = 1
    self.machines[3].width = 1
    self.machines[3].pos_input = [7, 4]
    self.machines[3].pos_output = [7, 4]
    self.machines[3].input_products = ['m3_m4']
    self.machines[3].output_products = ['m4_m2']
    self.machines[3].factory = self'''

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


def create_default_factory_5(self):
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

    print(self.product_types)

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

    print(self.product_types)

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
    self.machines[1].output_products = ['default_product_1'] #todo hier vorher "empty, war jedoch nicht in product_types drin --> change product hat nicht funktioniert
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
    self.machines[2].output_products = ['default_product_1'] #todo vorher empty
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


def not_working_factory_1(self):
    self.length = 20
    self.width = 10
    self.no_columns = int(self.length // self.cell_size)
    self.no_rows = int(self.width // self.cell_size)
    self.factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows)).tolist()

    self.product_types['0'] = dict(length=800, width=800, weight=1)
    self.product_types['1'] = dict(length=250, width=250, weight=1)
    self.product_types['2'] = dict(length=1200, width=800, weight=1)
    self.product_types['3'] = dict(length=1200, width=800, weight=1)

    print(self.product_types)

    self.warehouses.append(Warehouse())
    self.warehouses[0].name = 'W0'
    self.warehouses[0].pos_x = 12
    self.warehouses[0].pos_y = 8
    self.warehouses[0].length = 2
    self.warehouses[0].width = 2
    self.warehouses[0].pos_input = [12, 8]
    self.warehouses[0].pos_output = [13, 8]
    self.warehouses[0].input_products = ['1']
    self.warehouses[0].output_products = ['3']
    self.warehouses[0].factory = self

    self.machines.append(Machine())
    self.machines[0].name = 'M0'
    self.machines[0].pos_x = 12
    self.machines[0].pos_y = 0
    self.machines[0].length = 2
    self.machines[0].width = 2
    self.machines[0].pos_input = [12, 0]
    self.machines[0].pos_output = [13, 0]
    self.machines[0].input_products = ['3']
    self.machines[0].output_products = ['1']
    self.machines[0].factory = self

    self.machines.append(Machine())
    self.machines[1].name = 'M1'
    self.machines[1].pos_x = 0
    self.machines[1].pos_y = 8
    self.machines[1].length = 2
    self.machines[1].width = 2
    self.machines[1].pos_input = [0, 8]
    self.machines[1].pos_output = [1, 8]
    self.machines[1].input_products = ['0']
    self.machines[1].output_products = ['2']
    self.machines[1].factory = self

    self.machines.append(Machine())
    self.machines[2].name = 'M2'
    self.machines[2].pos_x = 6
    self.machines[2].pos_y = 4
    self.machines[2].length = 2
    self.machines[2].width = 2
    self.machines[2].pos_input = [6, 4]
    self.machines[2].pos_output = [7, 4]
    self.machines[2].input_products = ['2']
    self.machines[2].output_products = ['0']
    self.machines[2].factory = self

    self.loading_stations.append(LoadingStation())
    self.loading_stations[0].name = f'L0'
    self.loading_stations[0].pos_x = 14
    self.loading_stations[0].pos_y = 5
    self.loading_stations[0].length = 1
    self.loading_stations[0].width = 1
    self.agvs.append(AGV([14, 5]))
    self.agvs[0].name = 'AGV0'
    self.agvs[0].thread_running = False
    self.agvs[0].factory = self

    self.loading_stations.append(LoadingStation())
    self.loading_stations[1].name = f'L1'
    self.loading_stations[1].pos_x = 10
    self.loading_stations[1].pos_y = 9
    self.loading_stations[1].length = 1
    self.loading_stations[1].width = 1
    self.agvs.append(AGV([10, 9]))
    self.agvs[1].name = 'AGV1'
    self.agvs[1].thread_running = False
    self.agvs[1].factory = self

    self.loading_stations.append(LoadingStation())
    self.loading_stations[2].name = f'L2'
    self.loading_stations[2].pos_x = 1
    self.loading_stations[2].pos_y = 1
    self.loading_stations[2].length = 1
    self.loading_stations[2].width = 1
    self.agvs.append(AGV([1, 1]))
    self.agvs[2].name = 'AGV2'
    self.agvs[2].thread_running = False
    self.agvs[2].factory = self

    self.loading_stations.append(LoadingStation())
    self.loading_stations[3].name = f'L3'
    self.loading_stations[3].pos_x = 6
    self.loading_stations[3].pos_y = 6
    self.loading_stations[3].length = 1
    self.loading_stations[3].width = 1
    self.agvs.append(AGV([6, 6]))
    self.agvs[3].name = 'AGV3'
    self.agvs[3].thread_running = False
    self.agvs[3].factory = self

    self.loading_stations.append(LoadingStation())
    self.loading_stations[4].name = f'L4'
    self.loading_stations[4].pos_x = 18
    self.loading_stations[4].pos_y = 6
    self.loading_stations[4].length = 1
    self.loading_stations[4].width = 1
    self.agvs.append(AGV([18, 6]))
    self.agvs[4].name = 'AGV4'
    self.agvs[4].thread_running = False
    self.agvs[4].factory = self

    self.loading_stations.append(LoadingStation())
    self.loading_stations[5].name = f'L5'
    self.loading_stations[5].pos_x = 18
    self.loading_stations[5].pos_y = 9
    self.loading_stations[5].length = 1
    self.loading_stations[5].width = 1
    self.agvs.append(AGV([18, 9]))
    self.agvs[5].name = 'AGV5'
    self.agvs[5].thread_running = False
    self.agvs[5].factory = self

    self.fill_grid()

def create_temp_factory_machines_PAPER(self):
    self.length = 25
    self.width = 15
    self.no_columns = int(self.length // self.cell_size)
    self.no_rows = int(self.width // self.cell_size)
    self.factory_grid_layout = np.zeros(shape=(self.no_columns, self.no_rows)).tolist()

    self.product_types['W0_Output'] = dict(length=1200, width=900, weight=5)
    self.product_types['M0_Output'] = dict(length=400, width=400, weight=5)
    self.product_types['M1_Output'] = dict(length=250, width=250, weight=5)
    self.product_types['M5_Output'] = dict(length=800, width=800, weight=5)
    self.product_types['M6_Output'] = dict(length=400, width=400, weight=5)


    self.warehouses.append(Warehouse())
    self.warehouses[0].name = 'W0'
    self.warehouses[0].pos_x = 0
    self.warehouses[0].pos_y = 6
    self.warehouses[0].length = 4
    self.warehouses[0].width = 9
    self.warehouses[0].pos_input = [3, 13]
    self.warehouses[0].pos_output = [3, 10]
    self.warehouses[0].input_products = ['a']
    self.warehouses[0].output_products = ['b']
    self.warehouses[0].process_time = 10
    self.warehouses[0].rest_process_time = 10
    self.warehouses[0].factory = self

    self.machines.append(Machine())
    self.machines[0].name = 'M0'
    self.machines[0].pos_x = 0
    self.machines[0].pos_y = 0
    self.machines[0].length = 4
    self.machines[0].width = 6
    self.machines[0].pos_input = [3, 5]
    self.machines[0].pos_output = [3, 3]
    self.machines[0].input_products = ['W0_Output']
    self.machines[0].output_products = ['M0_Output']
    self.machines[0].process_time = 60
    self.machines[0].rest_process_time = 60
    self.machines[0].factory = self

    self.machines.append(Machine())
    self.machines[1].name = 'M1'
    self.machines[1].pos_x = 5
    self.machines[1].pos_y = 4
    self.machines[1].length = 4
    self.machines[1].width = 2
    self.machines[1].pos_input = [5, 4]
    self.machines[1].pos_output = [8, 4]
    self.machines[1].input_products = ['M0_Output']
    self.machines[1].output_products = ['M1_Output']
    self.machines[1].process_time = 15
    self.machines[1].rest_process_time = 15
    self.machines[1].factory = self

    self.machines.append(Machine())
    self.machines[2].name = 'M2'
    self.machines[2].pos_x = 10
    self.machines[2].pos_y = 4
    self.machines[2].length = 4
    self.machines[2].width = 2
    self.machines[2].pos_input = [10, 4]
    self.machines[2].pos_output = [13, 4]
    self.machines[2].input_products = ['M0_Output']
    self.machines[2].output_products = ['M1_Output']
    self.machines[2].process_time = 15
    self.machines[2].rest_process_time = 15
    self.machines[2].factory = self

    self.machines.append(Machine())
    self.machines[3].name = 'M3'
    self.machines[3].pos_x = 15
    self.machines[3].pos_y = 4
    self.machines[3].length = 4
    self.machines[3].width = 2
    self.machines[3].pos_input = [15, 4]
    self.machines[3].pos_output = [18, 4]
    self.machines[3].input_products = ['M0_Output']
    self.machines[3].output_products = ['M1_Output']
    self.machines[3].process_time = 15
    self.machines[3].rest_process_time = 15
    self.machines[3].factory = self

    self.machines.append(Machine())
    self.machines[4].name = 'M4'
    self.machines[4].pos_x = 15
    self.machines[4].pos_y = 1
    self.machines[4].length = 4
    self.machines[4].width = 2
    self.machines[4].pos_input = [15, 2]
    self.machines[4].pos_output = [18, 2]
    self.machines[4].input_products = ['M0_Output']
    self.machines[4].output_products = ['M1_Output']
    self.machines[4].process_time = 15
    self.machines[4].rest_process_time = 15
    self.machines[4].factory = self

    self.machines.append(Machine())
    self.machines[5].name = 'M5'
    self.machines[5].pos_x = 20
    self.machines[5].pos_y = 0
    self.machines[5].length = 5
    self.machines[5].width = 15
    self.machines[5].pos_input = [20, 3]
    self.machines[5].pos_output = [20, 10]
    self.machines[5].input_products = ['c']
    self.machines[5].output_products = ['d']
    self.machines[5].process_time = 60
    self.machines[5].rest_process_time = 60
    self.machines[5].factory = self

    self.machines.append(Machine())
    self.machines[6].name = 'M6'
    self.machines[6].pos_x = 10
    self.machines[6].pos_y = 6
    self.machines[6].length = 4
    self.machines[6].width = 4
    self.machines[6].pos_input = [13, 9]
    self.machines[6].pos_output = [10, 9]
    self.machines[6].input_products = ['e']
    self.machines[6].output_products = ['f']
    self.machines[6].process_time = 30
    self.machines[6].rest_process_time = 30
    self.machines[6].factory = self

    self.machines.append(Machine())
    self.machines[7].name = 'M7'
    self.machines[7].pos_x = 10
    self.machines[7].pos_y = 11
    self.machines[7].length = 4
    self.machines[7].width = 4
    self.machines[7].pos_input = [13, 11]
    self.machines[7].pos_output = [10, 11]
    self.machines[7].input_products = ['g']
    self.machines[7].output_products = ['h']
    self.machines[7].process_time = 30
    self.machines[7].rest_process_time = 30
    self.machines[7].factory = self

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

    self.fill_grid()

    self.factory_grid_layout[9][2] = Path()
    for x in range(6):
        self.factory_grid_layout[x+STARTING_POINT][1] = Path()
    for y in range(13):
        self.factory_grid_layout[4][y+2] = Path()
    for x in range(15):
        self.factory_grid_layout[x+5][3] = Path()
    for y in range(8):
        self.factory_grid_layout[19][y+3] = Path()
    for x in range(10):
        self.factory_grid_layout[x+9][10] = Path()
    for y in range(12):
        self.factory_grid_layout[9][y+3] = Path()
    for x in range(4):
        self.factory_grid_layout[x+5][14] = Path()
    for y in range(7):
        self.factory_grid_layout[14][y+3] = Path()


def make_random_factory():

    random_factory = VRP_Simulation.random_factory_generator.create_random_factory()

    print('###########################################')
    print('##### Creating random factory for VRP #####')
    print('###########################################')

    random_factory.fill_grid()
    return random_factory

def make_random_grid_factory_with_paths():

    random_factory = VRP_Simulation.random_grid_factory_with_paths.create_random_factory()

    print('###########################################')
    print('##### Creating random factory for VRP #####')
    print('###########################################')

    return random_factory


def main():
    # factory = Factory()

    # factory.create_temp_factory_machines()
    # factory.create_temp_factory_machines_deadlock()
    # factory.create_temp_factory_machines_4()
    # factory.create_temp_factory_machines_PAPER()
    # factory.create_temp_factory_machines_3()

    # create_default_factory(factory)  # einfaches Beispiel
    # create_default_factory_2(factory)  # Gitterrostproduktion
    # create_default_factory_3(factory)  # einfaches Beispiel
    # create_default_factory_4(factory)  # einfaches Beispiel
    # create_default_factory_5(factory)  # einfaches Beispiel
    # create_default_logistic_environment(factory)
    # create_temp_factory_machines_PAPER(factory)

    # create_random_logistic_environment(factory)
    # create_random_factory(factory)  # randomisierte Fabrik mit Warenhäusern und Maschinen

    # factory = make_random_factory()
    # factory = make_random_grid_factory_with_paths()
    # not_working_factory_1(factory)

    for num_vehicles in range(13):
        if num_vehicles >= 0:

            with open(
                    f"C:/Users/mente/PycharmProjects/ZellFTF_2DSim_PAPER_OR/VRP_Simulation/Random_Path_Factories/20250108_10-43-46_random_factory.pkl",
                    'rb') as inp:
                factory = pickle.load(inp)

            if num_vehicles >= 1:
                factory.loading_stations.append(LoadingStation())
                factory.loading_stations[6].pos_x = 3
                factory.loading_stations[6].pos_y = 5
                factory.loading_stations[6].length = 1
                factory.loading_stations[6].width = 1
                factory.agvs.append(AGV(start_position=[3, 5], static=True))
                factory.agvs[6].name = 'AGV_6'
                factory.agvs[6].factory = factory
                factory.add_to_grid(factory.loading_stations[6])

            if num_vehicles >= 2:
                factory.loading_stations.append(LoadingStation())
                factory.loading_stations[7].pos_x = 11
                factory.loading_stations[7].pos_y = 1
                factory.loading_stations[7].length = 1
                factory.loading_stations[7].width = 1
                factory.agvs.append(AGV(start_position=[11, 1], static=True))
                factory.agvs[7].name = 'AGV_7'
                factory.agvs[7].factory = factory
                factory.add_to_grid(factory.loading_stations[7])

            if num_vehicles >= 3:
                factory.loading_stations.append(LoadingStation())
                factory.loading_stations[8].pos_x = 19
                factory.loading_stations[8].pos_y = 1
                factory.loading_stations[8].length = 1
                factory.loading_stations[8].width = 1
                factory.agvs.append(AGV(start_position=[19, 1], static=True))
                factory.agvs[8].name = 'AGV_8'
                factory.agvs[8].factory = factory
                factory.add_to_grid(factory.loading_stations[8])

            if num_vehicles >= 4:
                factory.loading_stations.append(LoadingStation())
                factory.loading_stations[9].pos_x = 3
                factory.loading_stations[9].pos_y = 9
                factory.loading_stations[9].length = 1
                factory.loading_stations[9].width = 1
                factory.agvs.append(AGV(start_position=[3, 9], static=True))
                factory.agvs[9].name = 'AGV_9'
                factory.agvs[9].factory = factory
                factory.add_to_grid(factory.loading_stations[9])

            if num_vehicles >= 5:
                factory.loading_stations.append(LoadingStation())
                factory.loading_stations[10].pos_x = 15
                factory.loading_stations[10].pos_y = 5
                factory.loading_stations[10].length = 1
                factory.loading_stations[10].width = 1
                factory.agvs.append(AGV(start_position=[15, 5], static=True))
                factory.agvs[10].name = 'AGV_10'
                factory.agvs[10].factory = factory
                factory.add_to_grid(factory.loading_stations[10])

            if num_vehicles >= 6:
                factory.loading_stations.append(LoadingStation())
                factory.loading_stations[11].pos_x = 7
                factory.loading_stations[11].pos_y = 9
                factory.loading_stations[11].length = 1
                factory.loading_stations[11].width = 1
                factory.agvs.append(AGV(start_position=[7, 9], static=True))
                factory.agvs[11].name = 'AGV_11'
                factory.agvs[11].factory = factory
                factory.add_to_grid(factory.loading_stations[11])

            if num_vehicles >= 7:
                factory.loading_stations.append(LoadingStation())
                factory.loading_stations[12].pos_x = 3
                factory.loading_stations[12].pos_y = 7
                factory.loading_stations[12].length = 1
                factory.loading_stations[12].width = 1
                factory.agvs.append(AGV(start_position=[3, 7], static=True))
                factory.agvs[12].name = 'AGV_12'
                factory.agvs[12].factory = factory
                factory.add_to_grid(factory.loading_stations[12])

            if num_vehicles >= 8:
                factory.loading_stations.append(LoadingStation())
                factory.loading_stations[13].pos_x = 11
                factory.loading_stations[13].pos_y = 3
                factory.loading_stations[13].length = 1
                factory.loading_stations[13].width = 1
                factory.agvs.append(AGV(start_position=[11, 3], static=True))
                factory.agvs[13].name = 'AGV_13'
                factory.agvs[13].factory = factory
                factory.add_to_grid(factory.loading_stations[13])

            if num_vehicles >= 9:
                factory.loading_stations.append(LoadingStation())
                factory.loading_stations[14].pos_x = 19
                factory.loading_stations[14].pos_y = 3
                factory.loading_stations[14].length = 1
                factory.loading_stations[14].width = 1
                factory.agvs.append(AGV(start_position=[19, 3], static=True))
                factory.agvs[14].name = 'AGV_14'
                factory.agvs[14].factory = factory
                factory.add_to_grid(factory.loading_stations[14])

            if num_vehicles >= 10:
                factory.loading_stations.append(LoadingStation())
                factory.loading_stations[15].pos_x = 3
                factory.loading_stations[15].pos_y = 11
                factory.loading_stations[15].length = 1
                factory.loading_stations[15].width = 1
                factory.agvs.append(AGV(start_position=[3, 11], static=True))
                factory.agvs[15].name = 'AGV_15'
                factory.agvs[15].factory = factory
                factory.add_to_grid(factory.loading_stations[15])

            if num_vehicles >= 11:
                factory.loading_stations.append(LoadingStation())
                factory.loading_stations[16].pos_x = 15
                factory.loading_stations[16].pos_y = 7
                factory.loading_stations[16].length = 1
                factory.loading_stations[16].width = 1
                factory.agvs.append(AGV(start_position=[15, 7], static=True))
                factory.agvs[16].name = 'AGV_16'
                factory.agvs[16].factory = factory
                factory.add_to_grid(factory.loading_stations[16])

            if num_vehicles >= 12:
                factory.loading_stations.append(LoadingStation())
                factory.loading_stations[17].pos_x = 7
                factory.loading_stations[17].pos_y = 11
                factory.loading_stations[17].length = 1
                factory.loading_stations[17].width = 1
                factory.agvs.append(AGV(start_position=[7, 11], static=True))
                factory.agvs[17].name = 'AGV_17'
                factory.agvs[17].factory = factory
                factory.add_to_grid(factory.loading_stations[17])

            for j in range(1):
                filename = os.path.basename(__file__)[:-3]
                file = open(f"{filename}_POOL06_transport_duration.txt", "a")
                print(filename)

                vrp = VRP_cellAGV(factory)

                vrp.create_file_for_list_of_factory_objects()
                vrp.create_dataframe_of_factory_objects()
                # D = vrp.get_distance_matrix()
                D = factory.path_graph.get_object_distance_matrix(only_free_agv=True)
                # vrp.get_delivery_relationship()
                # Q = vrp.get_amount_of_agv_for_delivery_as_matrix_free_configuration()  # Konfiguration der FTF
                Q = vrp.get_amount_of_agv_for_delivery_as_matrix_1_4_6_configuration()  # Konfiguration der FTF
                machine_times = factory.get_idle_process_times()

                vrp.cVRPPDmDnR(num_AGVs=len(factory.agvs), D=D, Q=Q, machine_times=machine_times)
                routing = vrp.get_order_of_transport_new()
                num_AGVS = 6 + num_vehicles

                print(f"Anzahl AGVs : {num_AGVS}")
                print(f"{j + 1} Simulation (Duration und Fitness):")
                print(f"DAUER:\n{str(vrp.duration)}")
                print(f"FITNESS:\n{str(vrp.fitness)}")
                print(f"ROUTING:\n{str(routing)}\n")
                # print(f"Distanzmatrix = \n{D.tolist()}")
                # print(table_to_gams.generate_gams_table(D))
                # print(f"\nKonfigurationsmatrix = \n{Q}")
                # print(table_to_gams.generate_gams_table(Q))


                file.write(f"ANZAHL AN AGVS: {str(num_AGVS)}\n")
                file.write(f"{j+1} Simulation (Duration und Fitness):\n")
                file.write(f"DAUER:\n{str(vrp.duration)}\n")
                file.write(f"FITNESS:\n{str(vrp.fitness)}\n")
                file.write(f"MIP GAP:\n{vrp.mip_gap}\n")
                file.write(f"ROUTING:\n{str(routing)}\n\n")

                file.close()

                # vrp.get_order_of_transport_new()

                # vrp.reconstruct_path()


if __name__ == '__main__':
    main()

"""
Notizen:
- Transportkostensätze darstellen durch Matrixmultiplikation (Menge AGV * Distanz)
- Zeitabhängigkeit reinbringen: immer wenn ein Produkt neu angeboten/nachgefragt wird, Modell neu durchrechnen?!
"""

