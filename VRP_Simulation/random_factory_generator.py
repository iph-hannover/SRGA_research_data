import random
import time

import numpy as np

from FactoryObjects.Factory import Factory
from FactoryObjects.Machine import Machine
from FactoryObjects.Warehouse import Warehouse
from FactoryObjects.LoadingStation import LoadingStation
from FactoryObjects.AGV import AGV


random_factory = Factory()
random_factory.length = 20
random_factory.width = 10
random_factory.cell_size = 1
random_factory.no_columns = int(random_factory.length / random_factory.cell_size)
random_factory.no_rows = int(random_factory.width / random_factory.cell_size)
random_factory.np_factory_grid_layout = np.zeros(shape=(random_factory.no_columns, random_factory.no_rows))
random_factory.factory_grid_layout = random_factory.np_factory_grid_layout.tolist()
amount_of_machines = 4
amount_of_warehouses = 1
amount_of_agvs = 6
amount_of_products = 5


def create_random_products():
    random_products = []
    random_products.append(dict(length=250, width=250, weight=1))  # 1 FTF
    random_products.append(dict(length=600, width=600, weight=4))  # 4 FTF
    random_products.append(dict(length=1100, width=600, weight=6))  # 6 FTF
    for i in range(amount_of_products):
        random_factory.product_types[str(i)] = random.choice(random_products)
    random_factory.product_types.pop("default_product_1")
    random_factory.product_types.pop("default_product_2")
    random_factory.product_types.pop("default_product_3")
    random_factory.product_types.pop("default_product_4")


def create_random_objects():
    print(random_factory.product_types)
    print(random_factory.product_types.keys())
    random_input_products = random_factory.product_types.copy()
    random_output_products = random_factory.product_types.copy()
    random_x_positions_for_objects = [x for x in range(random_factory.length) if x % 2 == 0]
    random_y_positions_for_objects = [y for y in range(random_factory.width) if y % 2 == 0]
    random_object_positions = []
    for i in random_x_positions_for_objects:
        for j in random_y_positions_for_objects:
            random_object_positions.append([i, j])
    random_x_positions_for_agvs = [x for x in range(random_factory.length)]
    random_y_positions_for_agvs = [y for y in range(random_factory.width)]
    random_agv_positions = []
    for i in random_x_positions_for_agvs:
        for j in random_y_positions_for_agvs:
            random_agv_positions.append([i, j])
    for i in range(amount_of_machines):
        random_factory.machines.append(Machine())
        random_factory.machines[i].name = f'Maschine_{i}'

        random_position = random.choice(random_object_positions)
        random_x_position = random_position[0]
        random_y_position = random_position[1]
        random_factory.machines[i].pos_x = random_x_position
        random_factory.machines[i].pos_y = random_y_position
        random_object_positions.remove([random_x_position, random_y_position])
        random_agv_positions.remove([random_x_position, random_y_position])
        random_agv_positions.remove([random_x_position + 1, random_y_position])

        random_factory.machines[i].length = 2
        random_factory.machines[i].width = 2
        random_factory.machines[i].pos_input = [random_factory.machines[i].pos_x, random_factory.machines[i].pos_y]
        random_factory.machines[i].pos_output = [random_factory.machines[i].pos_x + 1, random_factory.machines[i].pos_y]
        chosen_input_product = random.choice(list(random_input_products.keys()))
        chosen_output_product = random.choice([x for x in list(random_output_products.keys()) if x!= chosen_input_product])
        random_factory.machines[i].input_products = [chosen_input_product]
        random_factory.machines[i].output_products = [chosen_output_product]
        random_input_products.pop(chosen_input_product)
        random_output_products.pop(chosen_output_product)
        random_factory.machines[i].factory = random_factory
        process_time = random.randint(0,60)
        random_factory.machines[i].process_time = process_time
        random_factory.machines[i].rest_process_time = process_time

    for i in range(amount_of_warehouses):
        random_factory.warehouses.append(Warehouse())
        random_factory.warehouses[i].name = f'Warehouse_{i}'

        random_position = random.choice(random_object_positions)
        random_x_position = random_position[0]
        random_y_position = random_position[1]
        random_factory.warehouses[i].pos_x = random_x_position
        random_factory.warehouses[i].pos_y = random_y_position
        random_object_positions.remove([random_x_position, random_y_position])
        random_agv_positions.remove([random_x_position, random_y_position])
        random_agv_positions.remove([random_x_position + 1, random_y_position])

        random_factory.warehouses[i].length = 2
        random_factory.warehouses[i].width = 2
        random_factory.warehouses[i].pos_input = [random_factory.warehouses[i].pos_x, random_factory.warehouses[i].pos_y]
        random_factory.warehouses[i].pos_output = [random_factory.warehouses[i].pos_x + 1, random_factory.warehouses[i].pos_y]
        chosen_input_product = random.choice(list(random_input_products.keys()))
        chosen_output_product = random.choice(
            [x for x in list(random_output_products.keys()) if x != chosen_input_product])
        random_factory.warehouses[i].input_products = [chosen_input_product]
        random_factory.warehouses[i].output_products = [chosen_output_product]
        random_input_products.pop(chosen_input_product)
        random_output_products.pop(chosen_output_product)
        random_factory.warehouses[i].factory = random_factory
        process_time = random.randint(0, 60)
        random_factory.warehouses[i].process_time = process_time
        random_factory.warehouses[i].rest_process_time = process_time

    for i in range(amount_of_agvs):
        random_factory.loading_stations.append(LoadingStation())

        random_position = random.choice(random_agv_positions)
        random_x_position = random_position[0]
        random_y_position = random_position[1]
        random_factory.loading_stations[i].pos_x = random_x_position
        random_factory.loading_stations[i].pos_y = random_y_position
        random_factory.loading_stations[i].length = 1
        random_factory.loading_stations[i].width = 1
        random_factory.agvs.append(AGV([random_x_position, random_y_position]))
        random_factory.agvs[i].name = f'AGV_{i}'
        random_factory.agvs[i].factory = random_factory

        random_agv_positions.remove([random_x_position, random_y_position])

def create_random_factory():
    create_random_products()
    create_random_objects()

    return random_factory


