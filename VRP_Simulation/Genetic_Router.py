import copy
import random
import math
import sys
import logging
import numpy as np
import pandas as pd

from FactoryObjects.AGV import AGV
from FactoryObjects.Factory import Factory
from FactoryObjects.Warehouse import Warehouse
from FactoryObjects.Machine import Machine


class GeneticRouter:
    def __init__(self, factory: Factory, population_size: int = 10, generations: int = 10, mutation_rate: float = 0.4,
                 swap_percentage_crossover: float = 0.4, fitness_rate: float = 0.1, fitness_weight: float = 1,
                 use_paths=False, write_data=False, with_prints=True):
        self.factory = factory
        self.amount_of_nodes = self.factory.get_amount_of_delivery_nodes()
        self.dimension = self.amount_of_nodes
        self.list_of_factory_objects_input_output = (
            self.factory.get_list_of_factory_objects_agvs_warehouse_machines_input_output())
        self.list_agv_input_output = self.factory.get_list_input_output()
        self.agv = AGV()
        self.agv.thread_running = False
        self.agv.length = 500  # length of agv in mm
        self.agv.width = 500  # width of agv in mm
        self.project_path = sys.path[1]
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.swap_percentage_crossover = swap_percentage_crossover
        self.fitness_rate = fitness_rate
        self.w1 = fitness_weight
        self.w2 = 1 - self.w1
        self.agv_available_idx = self.factory.get_available_agvs_idx()
        print(f"AGVS AVAILABLE: {self.agv_available_idx}")
        self.amount_agvs = len(self.agv_available_idx)
        self.num_locations = 0
        self.D = None
        self.Q = None

        self.use_paths = use_paths

        self.logger = logging.getLogger()
        if with_prints:
            self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)


    def get_routing(self):
        if self.use_paths:
            self.D = self.factory.path_graph.get_object_distance_matrix(only_free_agv=True)
        else:
            self.D = self._get_distance_matrix()
        # self.D = self._get_distance_matrix()
        self.Q = self.get_amount_of_agv_for_delivery_as_matrix_1_4_6_configuration()
        self.num_locations = len(self.D)
        routing = self.genetic_algorithm()
        return routing

    def create_list_of_factory_objects(self):
        """
        This function writes all factory objects (Warehouses, Machines, AGVs) to a list.
        :return: list_of_factory_objects_name
                 A list of all factory objects in the order mentioned above as a list
        """
        list_of_factory_objects_name = []
        for i in range(self.dimension):
            if isinstance(self.list_of_factory_objects_input_output[i], AGV):
                list_of_factory_objects_name.append(self.list_of_factory_objects_input_output[i].name)
            else:
                if self.list_of_factory_objects_input_output[i] != self.list_of_factory_objects_input_output[i - 1]:
                    list_of_factory_objects_name.append(f'{self.list_of_factory_objects_input_output[i].name}_input')
                else:
                    list_of_factory_objects_name.append(f'{self.list_of_factory_objects_input_output[i].name}_output')
        # list_of_factory_objects_pd = pd.DataFrame(list_of_factory_objects_name)
        # list_of_factory_objects_pd.to_csv(self.project_path + '/data/Current_Factory/VRP/list_of_factory_objects.csv',
        #                                   sep=';')
        return list_of_factory_objects_name

    def _get_distance_matrix(self):
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
                if ((type(self.list_of_factory_objects_input_output[i]) == AGV and self.list_of_factory_objects_input_output[i].is_free) and
                        (type(self.list_of_factory_objects_input_output[j]) == AGV and self.list_of_factory_objects_input_output[j].is_free)):
                    distance_x = abs(self.list_of_factory_objects_input_output[i].pos_x
                                     - self.list_of_factory_objects_input_output[j].pos_x)
                    distance_y = abs(self.list_of_factory_objects_input_output[i].pos_y
                                     - self.list_of_factory_objects_input_output[j].pos_y)
                    # self.logger.debug(f'{self.list_of_factory_objects_input_output[i].name}: '
                    #       f'pos_x = {self.list_of_factory_objects_input_output[i].pos_x}, '
                    #       f'pos_y = {self.list_of_factory_objects_input_output[i].pos_y}')
                # 2nd if: AGV - Machine/Warehouse
                elif ((type(self.list_of_factory_objects_input_output[i]) == AGV and self.list_of_factory_objects_input_output[i].is_free) and
                      (type(self.list_of_factory_objects_input_output[j]) == Warehouse or
                       type(self.list_of_factory_objects_input_output[j]) == Machine)):
                    if self.list_agv_input_output[j] == 'input':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_x -
                                         self.list_of_factory_objects_input_output[j].pos_input[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_y -
                                         self.list_of_factory_objects_input_output[j].pos_input[1])
                    elif self.list_agv_input_output[j] == 'output':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_x -
                                         self.list_of_factory_objects_input_output[j].pos_output[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_y -
                                         self.list_of_factory_objects_input_output[j].pos_output[1])
                # 3rd if: Machine/Warehouse - AGV
                elif ((type(self.list_of_factory_objects_input_output[i]) == Warehouse or
                       type(self.list_of_factory_objects_input_output[i]) == Machine) and
                      (type(self.list_of_factory_objects_input_output[j]) == AGV and self.list_of_factory_objects_input_output[j].is_free)):
                    if self.list_agv_input_output[i] == 'input':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_input[0] -
                                         self.list_of_factory_objects_input_output[j].pos_x)
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_input[1] -
                                         self.list_of_factory_objects_input_output[j].pos_y)
                    elif self.list_agv_input_output[i] == 'output':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_output[0] -
                                         self.list_of_factory_objects_input_output[j].pos_x)
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_output[1] -
                                         self.list_of_factory_objects_input_output[j].pos_y)
                # 4th if: Machine/Warehouse - Machine/Warehouse
                elif ((type(self.list_of_factory_objects_input_output[i]) == Warehouse or
                       type(self.list_of_factory_objects_input_output[i]) == Machine) and
                      (type(self.list_of_factory_objects_input_output[i]) == Warehouse or
                       type(self.list_of_factory_objects_input_output[i]) == Machine)):
                    if self.list_agv_input_output[i] == 'input' and self.list_agv_input_output[j] == 'output':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_input[0] -
                                         self.list_of_factory_objects_input_output[j].pos_output[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_input[1] -
                                         self.list_of_factory_objects_input_output[j].pos_output[1])
                    elif self.list_agv_input_output[i] == 'input' and self.list_agv_input_output[j] == 'input':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_input[0] -
                                         self.list_of_factory_objects_input_output[j].pos_input[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_input[1] -
                                         self.list_of_factory_objects_input_output[j].pos_input[1])
                    elif self.list_agv_input_output[i] == 'output' and self.list_agv_input_output[j] == 'input':
                        distance_x = abs(self.list_of_factory_objects_input_output[i].pos_output[0] -
                                         self.list_of_factory_objects_input_output[j].pos_input[0])
                        distance_y = abs(self.list_of_factory_objects_input_output[i].pos_output[1] -
                                         self.list_of_factory_objects_input_output[j].pos_input[1])
                    elif self.list_agv_input_output[i] == 'output' and self.list_agv_input_output[j] == 'output':
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
        distance_matrix_pd.to_csv(self.project_path + '/data/Current_Factory/VRP/distance_matrix.csv', sep=';')
        return distance_matrix.tolist()

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
                    if self.list_agv_input_output[i] == 'output':
                        if (type(self.list_of_factory_objects_input_output[j]) == Machine or
                                type(self.list_of_factory_objects_input_output[j]) == Warehouse):
                            if self.list_agv_input_output[j] == 'input':
                                for output_product in self.list_of_factory_objects_input_output[i].output_products:
                                    for input_product in self.list_of_factory_objects_input_output[j].input_products:
                                        if output_product == input_product:
                                            if self.list_of_factory_objects_input_output[i].has_product(output_product):
                                                self.logger.debug(
                                                    '---------------------------------------------------------------------------')
                                                self.logger.debug(f'{m:03d}. Lieferbeziehung gefunden! '
                                                      f'Lieferung von {self.list_of_factory_objects_input_output[i].name} nach '
                                                      f'{self.list_of_factory_objects_input_output[j].name}, Geliefertes Produkt: {output_product}')
                                                if (type(self.list_of_factory_objects_input_output[j]) == Machine and
                                                        self.list_of_factory_objects_input_output[j].get_buffer_status()[0] != 0 or
                                                        self.list_of_factory_objects_input_output[j].status == 'idle'):
                                                    # Kann nur beliefert werden, wenn der Buffer von Machine[j] frei ist
                                                    delivery_matrix_with_agv[i][
                                                        j] = self.check_amount_of_agvs_for_transport_1_4_6_configuration(
                                                        self.factory.product_types[output_product])
                                                elif type(self.list_of_factory_objects_input_output[j]) == Warehouse:
                                                    delivery_matrix_with_agv[i][
                                                        j] = self.check_amount_of_agvs_for_transport_1_4_6_configuration(
                                                        self.factory.product_types[output_product])
                                                if self.check_amount_of_agvs_for_transport_1_4_6_configuration(
                                                        self.factory.product_types[output_product]) > len(self.factory.get_num_free_agv_for_vrp()):
                                                    delivery_matrix_with_agv[i][j] = 0
                                                    # self.delivery_matrix_with_agv[i][j] = delivery_matrix_with_agv[i][j]
                                                # else:
                                                #     delivery_matrix_with_agv[i][
                                                #         j] = self.check_amount_of_agvs_for_transport_1_4_6_configuration(
                                                #         self.factory.product_types[output_product])
                                                m += 1
        # delivery_matrix_with_agv_pd = pd.DataFrame(delivery_matrix_with_agv, columns=list_of_factory_objects_name,
        #                                            index=list_of_factory_objects_name)
        # delivery_matrix_with_agv_pd.to_csv(
        #     self.project_path + '/data/Current_Factory/VRP/delivery_matrix_with_agv.csv', sep=';')
        return delivery_matrix_with_agv.tolist()

    def genetic_algorithm(self):
        population, jobs = self.initialize_population()
        if population is None:
            return None

        initial_population = copy.deepcopy(population)

        # self.logger.debug(f"Initiale Population: {population}")

        # print(f'Initiale Population:')
        # print(f"{population}")

        best_fitness_per_generation = []

        # Variablen zum Speichern der besten Lösung aller Generationen
        best_overall_route = None
        best_overall_cost = float('inf')
        best_overall_duration = float('inf')
        best_overall_fitness = float('inf')

        for generation in range(self.generations):
            self.logger.debug(f"\nGeneration {generation + 1}")
            new_population = copy.deepcopy(population)

            # Berechne die Fitnesswerte der aktuellen Population
            fitness_values = self.calculate_fitness_population(population)

            # Selektion der besten Lösungen
            beste_routen = self.select_best_fitness(population, fitness_values)
            # beste_route = self.select_best_route_after_each_step(population, fitness_values)
            # (f"Beste Route = {beste_route}")
            # self.logger.debug(f"Beste Routen = {beste_routen}")
            # self.logger.debug(f"LÄNGE Beste Routen = {len(beste_routen)}")

            # Crossover1 mit den besten Routen
            population_nach_crossover1 = []
            # self.logger.debug(f"### PERFORM CROSSOVER 1 MIT DEN BESTEN ROUTEN ###")
            for i in range(len(population)):
                route_c1 = copy.deepcopy(beste_routen[i % len(beste_routen)])
                route_nach_crossover1 = self.perform_crossover(route_c1, jobs)
                population_nach_crossover1.append(copy.deepcopy(route_nach_crossover1))
                new_population.append(copy.deepcopy(route_nach_crossover1))
            # self.logger.debug(f"POP NACH C1 ({len(population_nach_crossover1)}) : {population_nach_crossover1}")

            # Crossover2 mit zufällig ausgewählten Routen
            alle_anderen_routen = [route for route in population if route not in beste_routen]
            population_nach_crossover2 = []
            # self.logger.debug(f"### PERFORM CROSSOVER 2 MIT ALLEN ANDEREN ROUTEN ###")
            if len(alle_anderen_routen) > 0:
                for i in range(len(population)):
                    route_c2 = copy.deepcopy(random.choice(alle_anderen_routen))
                    route_nach_crossover2 = self.perform_crossover(route_c2, jobs)
                    population_nach_crossover2.append(copy.deepcopy(route_nach_crossover2))
                    new_population.append(copy.deepcopy(route_nach_crossover2))
            # self.logger.debug(f"POP NACH C2 ({len(population_nach_crossover2)}) : {population_nach_crossover2}")

            # Mutation der besten Routen
            population_nach_mutation_beste_routen = []
            for i in range(len(population)):
                route_m0 = beste_routen[i % len(beste_routen)]
                mutierte_route = self.mutate_routes(route_m0, jobs)
                population_nach_mutation_beste_routen.append(copy.deepcopy(mutierte_route))
                new_population.append(copy.deepcopy(mutierte_route))

            # Mutation der Crossover1 Routen
            population_nach_mutation1 = []
            for route_m1 in population_nach_crossover1:
                mutierte_route = copy.deepcopy(self.mutate_routes(route_m1, jobs))
                population_nach_mutation1.append(copy.deepcopy(mutierte_route))
                new_population.append(copy.deepcopy(mutierte_route))

            # Mutation der Crossover2 Routen
            population_nach_mutation2 = []
            for route_m2 in population_nach_crossover2:
                mutierte_route = copy.deepcopy(self.mutate_routes(route_m2, jobs))
                population_nach_mutation2.append(copy.deepcopy(mutierte_route))
                new_population.append(copy.deepcopy(mutierte_route))

            # Update der Population
            fitness_new_population = self.calculate_fitness_population(new_population)
            population = self.create_new_population_afer_mutation(new_population, fitness_new_population)

            # Berechne die Fitnesswerte der neuen Population
            new_fitness_values = self.calculate_fitness_population(population)
            best_fitness = min(new_fitness_values)
            best_fitness_per_generation.append(best_fitness)

            # Ausgabe der besten Lösung dieser Generation
            best_solution_index = new_fitness_values.index(best_fitness)
            best_solution = population[best_solution_index]
            self.logger.debug(f"Beste Lösung in Generation {generation + 1} Fitnesswert: {best_fitness:.4f}")

            # Aktualisiere die beste Lösung aller Generationen
            best_solution_cost = self.calculate_total_route_cost(best_solution)
            best_solution_duration = self.calculate_total_process_time(best_solution)

            # Prüfen, ob die Route gültig ist und ob der Fitnesswert besser ist
            if best_solution is not None and best_fitness < best_overall_fitness:
                best_overall_route = best_solution
                best_overall_cost = best_solution_cost
                best_overall_duration = best_solution_duration
                best_overall_fitness = best_fitness

        # Ausgabe der besten Lösung aller Generationen
        if best_overall_route is not None:
            # self.logger.debug("\nBeste Lösung aller Generationen:")
            # self.logger.debug(f"Route: {best_overall_route}")
            # discard route order after next operation
            updated_routes = self.get_updated_routes(best_overall_route[0])
            # self.logger.debug(f"Updated Route: {updated_routes}")
            x_values = self.get_x_values(updated_routes)
            self.x_values = x_values
            # self.logger.debug(f"x_values: {x_values}")
            transport_order = self.get_order_of_transport_new(x_values)
            # self.logger.debug(f"Reihenfolge des Transports: {transport_order}")
            # self.logger.debug(type(best_overall_route))
            # self.logger.debug(best_overall_route[0])
            # self.logger.debug(type(best_overall_route[0]))
            # self.logger.debug(best_overall_route[0][0])
            # self.logger.debug(type(best_overall_route[0][0]))
            self.logger.debug(f"Distanzkosten: {best_overall_cost}")
            self.logger.debug(f"Prozessdauer: {best_overall_duration}")
            self.logger.debug(f"Fitnesswert: {best_overall_fitness}")
            # transport_order, endknoten = process_transport_paths(best_overall_route)
            # self.logger.debug(f"Reihenfolge des Transports: {transport_order}")
            # self.logger.debug(f"Endknoten: {endknoten}")
        else:
            self.logger.info("\nKeine gültige Lösung gefunden.")
            transport_order = False

        return transport_order

    def initialize_population(self):
        population = []
        job_list = [(i, j, int(self.Q[i][j])) for i in range(self.num_locations) for j in range(self.num_locations) if
                    self.Q[i][j] > 0]
        print(f"Jobs: {job_list}")
        if len(job_list) < 1:
            print("no Jobs")
            return None, job_list
        for _ in range(self.population_size):
            solution, job_order = self.initial_solution(job_list)
            population.append((solution, job_order))
        return population, job_order

    def initial_solution(self, job_list):
        """
        Erstellt eine initiale zufällige Lösung, bei der die Routen so aufgebaut werden,
        dass Deadlocks vermieden werden und die Fahrzeuge effizient genutzt werden.
        """
        # Initiale Routen, jede Route beginnt beim jeweiligen Depot
        initial_routes = [[(i, i)] for i in range(self.amount_agvs)]  # [(Fahrzeugnummer, Depot)],...[
        agv_index_mapping = {}
        counter = 0
        for agv in self.agv_available_idx:
            agv_index_mapping[agv] = counter
            counter += 1

        # print(f"AGV_INDEX_MAPPING = {agv_index_mapping}")

        random.shuffle(job_list)

        # Liste, um die Reservierung der Fahrzeuge zu verfolgen
        # vehicle_reservations = [None] * self.amount_agvs

        for job in job_list:
            i, j, vehicles_needed = job
            available_vehicles = [v for v in
                                  self.agv_available_idx]  # if vehicle_reservations[agv_index_mapping[v]] is None]
            random.shuffle(available_vehicles)
            # print(f"AVAILABLE VEHICLES = {available_vehicles}")

            # Bei Aufträgen mit mehreren Fahrzeugen (Q>1) sicherstellen, dass genügend Fahrzeuge verfügbar sind
            if vehicles_needed > 1:
                if len(available_vehicles) >= vehicles_needed:
                    # Reserviere die erforderlichen Fahrzeuge für diesen Auftrag
                    reserved_vehicles = available_vehicles[:vehicles_needed]
                    free_vehicles = list(set(available_vehicles).difference(reserved_vehicles))
                    # print(f"RESERVED VEHICLES = {reserved_vehicles}")
                    # print(f"FREE VEHICLES = {free_vehicles}")
                    max_position = max(
                        len(initial_routes[agv_index_mapping[v]]) for n, v in enumerate(reserved_vehicles))
                    # print(f"MAX POSITION = {max_position}")

                    # Weise den Job den reservierten Fahrzeugen zu
                    for n, v in enumerate(reserved_vehicles):
                        # Synchronisiere die Routen so, dass alle Fahrzeuge den Auftrag im gleichen Schritt ausführen
                        while len(initial_routes[agv_index_mapping[v]]) < max_position:
                            initial_routes[agv_index_mapping[v]].append((None, None))  # Placeholder für Wartezeiten
                        initial_routes[agv_index_mapping[v]].append((i, j))
                        # vehicle_reservations[agv_index_mapping[v]] = job

                    # Synchronisiere die freien AGVs, das gleiche Lieferungen im gleichen Schritt ablaufen
                    for n, v in enumerate(free_vehicles):
                        # Synchronisiere die Routen so, dass alle Fahrzeuge den Auftrag im gleichen Schritt ausführen
                        while len(initial_routes[agv_index_mapping[v]]) < max_position:
                            initial_routes[agv_index_mapping[v]].append((None, None))  # Placeholder für Wartezeiten
                        initial_routes[agv_index_mapping[v]].append((None, None))
                        # vehicle_reservations[agv_index_mapping[v]] = job

                    # Entsperre die Fahrzeuge nach der Zuweisung, da der Auftrag abgeschlossen ist
                    for v in reserved_vehicles:
                        # vehicle_reservations[agv_index_mapping[v]] = None
                        pass
            else:
                # Wenn nur ein Fahrzeug benötigt wird, wird es einem freien Fahrzeug zugewiesen
                if available_vehicles:
                    vehicle = available_vehicles[0]
                    initial_routes[agv_index_mapping[vehicle]].append((i, j))
                    for vehicle_not_in_use in available_vehicles[1:]:
                        initial_routes[agv_index_mapping[vehicle_not_in_use]].append(
                            (None, None))  # Placeholder für Wartezeiten
                    # vehicle_reservations[agv_index_mapping[vehicle]] = None

        # Entferne Platzhalter (None, None) aus den Routen
        # for route in initial_routes:
        #    route[:] = [job for job in route if job != (None, None)]

        return initial_routes, job_list

    def calculate_fitness_population(self, population):
        fitness_values = []
        for route in population:
            total_cost = self.calculate_total_route_cost(route)
            total_process_time = self.calculate_total_process_time(route)
            fitness = self.calculate_fitness(total_cost, total_process_time)
            fitness_values.append(fitness)
        return fitness_values

    def calculate_total_route_cost(self, route_with_order):
        """
        Berechnet die Gesamtkosten aller Routen basierend auf der Distanzmatrix D.

        :param routes: Eine Liste von Routen, wobei jede Route eine Liste von Aufträgen ist.
        :return: Die Gesamtkosten aller Routen.
        """

        def calculate_single_route_cost(route):
            """
            Berechnet die Kosten (Distanzen) für eine einzelne Route (eine Route eines Fahrzeugs).
            """
            total_distance = 0
            if len(route) < 2:
                return 0  # Wenn die Route weniger als zwei Punkte hat, gibt es keine Distanz zu berechnen

            # Iteriere über alle Paare in der Route und berechne die Distanzen
            current_position = route[0][0]
            for i, j in route[1:]:
                if i is not None:
                    total_distance += self.D[current_position][i]  # Distanz vom aktuellen Punkt zum nächsten Auftrag
                    current_position = i
                    total_distance += self.D[i][j]
                    current_position = j  # Aktualisiere die aktuelle Position auf den Endpunkt des Auftrags

            return total_distance

        # Gesamtkosten
        total_cost = 0
        routes = route_with_order[0]
        for index, route in enumerate(routes):
            route_cost = calculate_single_route_cost(route)
            # self.logger.debug(f"FTF{index} = {route_cost:.2f}") # (optional) Ausgabe der Kosten für die einzelne Route
            total_cost += route_cost

        self.logger.debug(f"Distanzkosten: {total_cost:.2f} für Route: {routes}")  # Ausgabe der Gesamtkosten
        return total_cost  # Rückgabe der Gesamtkosten

    def calculate_total_process_time(self, routes_with_order):
        """
        Berechnet die Gesamtdauer des Prozesses.
        Angenommen, das Zurücklegen einer Distanzeinheit dauert eine Zeiteinheit.
        """

        def calculate_single_route_time(route):
            """
            Berechnet die gesamte Reisezeit für eine einzelne Route.
            """
            total_time = 0
            if len(route) < 2:
                return 0

            current_position = route[0][0]
            for i, j in route[1:]:
                if i is not None:
                    total_time += self.D[current_position][i]  # Zeit von current_position nach i
                    total_time += self.D[i][j]  # Zeit von i nach j
                    current_position = j

            return total_time

        routes = routes_with_order[0]
        max_time = 0
        for route in routes:
            route_time = calculate_single_route_time(route)
            max_time = max(max_time, route_time)

        return round(max_time, 2)  # Rückgabe der maximalen Prozessdauer

    def calculate_fitness(self, distance, process_time):
        """
        Berechnet den Fitnesswert einer Lösung basierend auf Distanz und Prozessdauer.
        Gewichtungsfaktoren w1 (steht für Distanz) und w2 (steht für Prozessdauer) können angepasst werden.
        """
        fitness_value = self.w1 * distance + self.w2 * process_time
        return fitness_value

    def select_best_fitness(self, routes, fitness_values):
        """
        Wählt die besten Routen basierend auf ihrem Fitnesswert aus.
        """
        # Anzahl der auszuwählenden Routen basierend auf dem Prozentsatz
        num_to_select = math.ceil(len(routes) * self.fitness_rate)

        # Kombiniere die Routen und Fitnesswerte in eine Liste von Tupeln
        routes_with_fitness = list(zip(routes, fitness_values))

        # Sortiere die Routen basierend auf ihrem Fitnesswert in aufsteigender Reihenfolge
        routes_with_fitness.sort(key=lambda x: x[1])

        # Wähle die besten Routen aus
        beste_routen = [route for route, fitness in routes_with_fitness[:num_to_select]]

        return beste_routen

    def select_best_route_after_each_step(self, routes, fitness_values):
        # Kombiniere die Routen und Fitnesswerte in eine Liste von Tupeln
        routes_with_fitness = list(zip(routes, fitness_values))

        # Sortiere die Routen basierend auf ihrem Fitnesswert in aufsteigender Reihenfolge
        routes_with_fitness.sort(key=lambda x: x[1])
        print(routes_with_fitness)

        # Wähle die beste Route aus
        beste_route = [route for route, fitness in routes_with_fitness[:1]]
        # self.beste_route = beste_route

        return beste_route

    def create_new_population_afer_mutation(self, new_population, fitness_new_population):
        num_to_select = math.ceil(self.population_size * self.fitness_rate)
        # Kombiniere die Population und Fitnesswerte in eine Liste von Tupeln
        population_with_fitness = list(zip(new_population, fitness_new_population))

        # Sortiere die Routen basierend auf ihrem Fitnesswert in aufsteigender Reihenfolge
        population_with_fitness.sort(key=lambda x: x[1])
        # print(population_with_fitness)

        beste_population = [route for route, fitness in population_with_fitness[:num_to_select]]
        alle_anderen = [route for route in new_population if route not in beste_population]
        gesamte_population = [route for route in new_population]

        print(f"Beste Population {len(beste_population)}")
        print(f"Alle anderen {len(alle_anderen)}")
        if len(alle_anderen) >= (self.population_size - num_to_select):
            random_population = random.choices(alle_anderen, k=int(self.population_size - num_to_select))
        else:
            random_population = random.choices(gesamte_population, k=int(self.population_size - num_to_select))

        return beste_population + random_population

    ########## Crossover ##########
    def perform_crossover(self, routes, jobs):
        """
        This function swaps the sequences of deliveries across all vehicles.
        Taking all vehicles into account at the same time is necessary to ensure that the resulting routes remain valid.
        :param routes: List containing the routes of the individual vehicles
        :param jobs: List containing all job nodes including the required number of vehicles.
        :return: new vehicle routes after crossover
        """
        # self.logger.debug(f"PERFORM CROSSOVER - Fahrzeugrouten ({len(routes[0])}) = {routes[0]}")
        # self.logger.debug(f"PERFORM CROSSOVER - Jobs ({len(jobs)}) = {jobs}")
        # self.logger.debug(f"PERFORM CROSSOVER - Länge einer Route mit None = {len(routes[0][0])}")
        # self.logger.debug(f"PERFORM CROSSOVER - Crossover-Rate = {self.swap_percentage_crossover}")
        list_amount_of_jobs = [i for i in range(len(routes[0][0]))]
        list_amount_of_jobs.pop(0)
        # self.logger.debug(f"PERFORM CROSSOVER - list_amount_of_jobs = {list_amount_of_jobs}")
        random.shuffle(list_amount_of_jobs)
        # self.logger.debug(f"PERFORM CROSSOVER - shuffled list_amount_of_jobs = {list_amount_of_jobs}")
        if len(list_amount_of_jobs) > 1:
            amount_of_swaps = math.ceil(self.swap_percentage_crossover * len(list_amount_of_jobs) / 2)
        else:
            amount_of_swaps = 0

        for i in range(amount_of_swaps):
            jobs_to_swap = list_amount_of_jobs[2 * i:2 * i + 2]
            # self.logger.debug(f"PERFORM CROSSVOER - Jobs to Swap: {jobs_to_swap}")
            j = 0
            for route in routes[0]:
                # self.logger.debug(f"{j}: Route vor Crossover : {route}")
                route[jobs_to_swap[0]], route[jobs_to_swap[1]] = route[jobs_to_swap[1]], route[jobs_to_swap[0]]
                # self.logger.debug(f"Route nach Crossover : {route}")
                routes[0][j] = route
                j += 1

        # self.logger.debug(f"PERFORM CROSSOVER - Fahrzeugrouten nach Crossover ({len(routes[0])}) = {routes[0]}")
        return routes

    def mutate_routes(self, routes, jobs):
        """
        Diese Funktion tauscht FTF-Routen zwischen einer zufälligen Auswahl von Fahrzeugen (FTFs)
        innerhalb einer Lösung basierend auf einem vorgegebenen Prozentsatz (zwischen 0 und 1).
        :param routes:
        :param jobs:
        :return:
        """
        # self.logger.debug(f"PERFORM MUTATION - Fahrzeugrouten ({len(routes[0])}), {type(routes[0])} = {routes[0]}")
        # self.logger.debug(f"PERFORM MUTATION - Jobs ({len(jobs)}) = {jobs}")
        # self.logger.debug(f"PERFORM MUTATION - Mutation-Rate = {self.mutation_rate}")
        list_amount_of_vehicles = [i for i in range(len(routes[0]))]
        # self.logger.debug(f"PERFORM MUTATION - list_amount_of_vehicles = {list_amount_of_vehicles}")
        random.shuffle(list_amount_of_vehicles)
        # self.logger.debug(f"PERFORM MUTATION - shuffled list_amount_of_vehicles = {list_amount_of_vehicles}")
        if len(routes[0]) > 1:
            amount_of_mutations = math.ceil(len(routes[0]) * self.mutation_rate / 2)
        else:
            amount_of_mutations = 0
        # self.logger.debug(f"PERFORM MUTATION - Amount of Mutations = {amount_of_mutations}")

        for i in range(amount_of_mutations):
            routes_to_swap = list_amount_of_vehicles[2 * i:2 * i + 2]
            # self.logger.debug(f"PERFORM MUTATION - Routes to Swap: {routes_to_swap}")
            routes[0][routes_to_swap[0]][1:], routes[0][routes_to_swap[1]][1:] = routes[0][routes_to_swap[1]][1:], \
            routes[0][routes_to_swap[0]][1:]

        # self.logger.debug(f"PERFORM MUTATION - Fahrzeugrouten nach Mutation ({len(routes[0])}) = {routes[0]}")
        return routes

    def get_updated_routes(self, routes):
        """
        Diese Funktion erstellt die Entscheidungsvariablen, die den Wert 1 annehmen (in Anlehnung an das Optimierungsmodell)
        um aus den Werten die Transportreihenfolge bestimmen zu können.
        :param routes: Die beste Route, die der genetische Algorithmus berechnet hat.
        :return: x_values - Entscheidungsvariablen mit dem Wert 1
        """
        updated_routes = []

        # Deletes all "(None, None)" entries from the routes
        for route in routes:
            route[:] = [entry for entry in route if entry != (None, None)]

        # self.logger.debug(f"UPDATED ROUTE - ROUTES = {routes}")

        for route in routes:
            updated_route = []
            if len(route) > 1:
                for i in range(len(route) - 1):
                    if i != 0:
                        updated_route.append(route[i])
                    # Add the travel from the current node to the next one
                    if route[i][1] != route[i + 1][0]:
                        updated_route.append((route[i][1], route[i + 1][0]))
                updated_route.append(route[len(route) - 1])
            else:
                # If there is only one node, we don't need to modify it
                updated_route = route

            updated_routes.append(updated_route)

        return updated_routes

    def get_x_values(self, updated_routes):
        dict_x_values = dict()
        for route_index, route in enumerate(updated_routes):
            for tup in route:
                # Create a new tuple that includes the index of the route and assign value 1.0
                dict_x_values[(tup[0], tup[1], route_index)] = 1.0

        return dict_x_values

    def get_order_of_transport_new(self, x_values):
        dict_order = dict()  # in the dict "dict_order" the order shall be saved as a dict,
        #                      the step number is the key and the routes (saved as (i,j,k)) are the values
        dict_order, remaining_x_values, actual_node_position = self.get_first_step_of_transport(x_values=x_values)
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

        if self.get_next_step_of_transport(x_values, remaining_x_values_dict_per_step, dict_order,
                                      actual_node_position_dict_per_step, step) == False:
            self.logger.debug('There is no solution\n')
            return False
        else:
            dict_order_of_transport = dict_order
            self.logger.debug(f'Solution: {dict_order}')
            return dict_order

    def get_first_step_of_transport(self, x_values):
        '''
        This is the ride of the agv to the initial pick-up nodes
        :return:
        '''
        dict_order = dict()  # in the dict "dict_order" the order shall be saved as a dict,
        #                      the step number is the key and the routes (saved as (i,j,k)) are the values
        working_list_order = []  # temporary list, which contains each x(i,j,k) for each step of the order as tuples
        remaining_x_values = x_values.copy()  # saves all remaining nodes for the iteration
        actual_node_position = dict()  # saves the temporary node position

        for i, j, k in x_values.keys():
            if i == k:
                working_list_order.append((i, j, k))  # adds the paths as a tuple
                remaining_x_values.pop((i, j, k))  # deletes the tuple from the remaining x_values
                actual_node_position[k] = j  # sets the current node position fpr each vehicle k
            dict_order[0] = working_list_order  # adds the travelled edge to dict

        return dict_order, remaining_x_values, actual_node_position

    def get_next_step_of_transport(self, x_values, remaining_x_values_dict_per_step, dict_order,
                                   actual_node_position_dict_per_step, step):
        # self.logger.debug(remaining_x_values_dict_per_step)
        # If there are no x_values remaining, then a solution is found
        if remaining_x_values_dict_per_step[step] == {}:
            return True

        # Try all combinations possible transports as a candidate from the actual node position of the vehicles
        possible_vehicle_edges = []  # temporary list which contains all the possible edges when vehicle k is at node i
        for _ in range(self.amount_agvs):
            possible_vehicle_edges.append([])

        vehicle_ids = set(vehicle_id for _, _, vehicle_id in x_values.keys())
        remaining_visited_edges = dict()
        for k in vehicle_ids:
            remaining_visited_edges[k] = [(i, j) for i, j, vehicle_id in x_values.keys() if
                                          x_values[(i, j, vehicle_id)] == 1.0 and vehicle_id == k]

        for k in actual_node_position_dict_per_step[step].keys():
            possible_vehicle_edges[k] = [edge for edge in remaining_visited_edges[k] if
                                         edge[0] == actual_node_position_dict_per_step[step][k]]
            # self.logger.debug('##########')

        # self.logger.debug(f'Mögliche Kanten: {possible_vehicle_edges}')

        vehicles = 0
        necessary_vehicles_for_transport = self.count_tuple_combinations(remaining_x_values_dict_per_step[step])
        # self.logger.debug(f'Necessary vehicles for transport: {necessary_vehicles_for_transport}')
        possible_edges = []
        for i in range(len(self.D)):  # jeder Knoten i wird durchgegangen
            for k in actual_node_position_dict_per_step[step].values():
                if k == i:
                    vehicles += 1  # wird +1, wenn ein Fahrzeug sich am Knoten i befindet
            for key, value in necessary_vehicles_for_transport.items():
                if value <= vehicles and i == key[0]:
                    # self.logger.debug(f'Knoten = {i}, Fahrzeuge = {vehicles}')
                    # self.logger.debug(f'Mögliche Fahrt: {key}')
                    possible_edges.append(key)
            vehicles = 0
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
            # self.logger.debug(f'{step}, {len_dict_order}')
            if len(dict_order) > step:
                for i in range(len_dict_order, step):
                    dict_order.pop(i)
            # self.logger.debug(f'Ausgewählte Kante: {possible_edge}')
            # self.logger.debug(f'Verbleibende Kanten gesamt nach Fahrt: {remaining_x_values_dict_per_step[step + 1]}')
            # self.logger.debug(f'Aktuelle Reihenfolge des Transports: {dict_order}')
            # self.logger.debug(f'Positionen der Fahrzeuge nach Iteration {step}: {actual_node_position_dict_per_step[step + 1]}')
            # self.logger.debug(
            #     '\n#######################################################################################################\n')
            if self.get_next_step_of_transport(x_values, remaining_x_values_dict_per_step, dict_order,
                                          actual_node_position_dict_per_step, step + 1) == True:
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