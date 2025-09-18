import sys
import time
import pygame
import matplotlib
import matplotlib.pyplot as plt
from FactoryObjects.Factory import Factory
from FactoryObjects.AGV import AGV
import pandas as pd

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class Factory_Simulation():
    def __init__(self, factory=None, render=False):
        super(Factory_Simulation, self).__init__()
        self.agv_positioning = None
        self.coupling_command = False
        self.agv_couple_count = None
        self.coupling_master = None

        if factory is None:
            self.factory = Factory()
            self.factory.create_temp_factory_machines_PAPER()
            # self.factory.create_temp_factory_machines()
        else:
            self.factory = factory

        self.list_of_factory_objects_input_output = self.factory.get_list_of_factory_objects_agvs_warehouse_machines_input_output()

        self.step_counter = 0
        self.time_step = 0.1
        self.sleep_time = 0.0001

        self.end_product_count = []
        plt.ion()

        self.agvs_workload = []
        self.machines_workload = []
        self.system_stock_data = [0.0]
        self.system_transport_data = [0.0]
        self.average_system_transport_data_for_csv = [0.0]
        self.average_system_stock_data_for_csv = [0.0]
        self.system_transport_data_for_csv = [0.0]
        self.system_stock_data_for_csv = [0.0]
        for i in range(len(self.factory.agvs)):
            self.agvs_workload.append([0])
        for i in range(len(self.factory.machines)):
            self.machines_workload.append([0])
        self.machine_status_durations = dict()
        for m in self.factory.machines:
            # logs overall machine state
            self.machine_status_durations[m.name] = {"process": 0, "idle": 0, "blocked": 0}
        self.machine_idle_process_times = dict()
        for m in self.factory.machines:
            # logs temp machine state to minimize delay times for VRP
            self.machine_idle_process_times[m.name] = {"process": 0.0, "idle": 0.0, "blocked": 0.0}
        self.agv_status_durations = dict()
        for a in self.factory.agvs:
            self.agv_status_durations[a.name] = {"loaded": 0, "unloaded": 0}
        self.machine_states_log = []
        self.agv_states_log = []
        self.agv_commands_log = []

        self.n_agv = len(self.factory.agvs)
        self.n_agv_commands = 5

        self.pixel_size = 50
        self.width = self.factory.length
        self.height = self.factory.width
        self.reference_size = self.factory.cell_size * 1000
        self.screen = None
        self.render_graphics = render
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width * self.pixel_size, self.height * self.pixel_size))
            pygame.display.set_caption('Color Data Display')

        self.project_path = sys.path[1]

        self.write_files = False

    def reset(self):
        self.factory.reset()
        self.step_counter = 0
        return False

    def step(self, action):
        self.step_counter += 1
        self._perform_action(action)
        self._block_until_synchronized()
        self._simulate_factory_objects()
        self.collect_data()
        # if self.step_counter % 10000 == 0:
        #     self.plot_durations()
        return self._get_relevant_vrp_change()

    def _get_relevant_vrp_change(self):
        has_new_product = False
        for warehouse in self.factory.warehouses:
            if warehouse.get_has_new_input_product():
                has_new_product = True
            if warehouse.get_has_new_output_product():
                has_new_product = True

        for machine in self.factory.machines:
            if machine.get_has_new_input_product():
                has_new_product = True
            if machine.get_has_new_output_product():
                has_new_product = True
            # if len(machine.buffer_output_load) > 0:
            #     if machine.unload_process_buffer():
            #         has_new_product = True

        return has_new_product

    def close(self):
        self.factory.shout_down()
        self._plot_training(True)
        plt.ioff()
        plt.show()

    def _perform_action(self, action):
        if action is None or not action or len(action) < 2:
            return

        # print(self.factory.agvs[0].coupling_master)
        commands = action[1]
        # print(f'Commands = {commands}')

        node_list = self.factory.get_list_of_factory_objects_agvs_warehouse_machines_input_output()
        # print(f'Node List: {node_list}')

        for command in commands:
            self.environment_set_action(node_list[command[2]], node_list[command[0]], node_list[command[1]])
            # print(f'Command - AGV: {command[2]}, output: {command[0]}, input: {command[1]}')
            # print('######################################')
            # print(f'Command - AGV: {node_list[command[2]].name}, output: {node_list[command[0]].name}, input: {node_list[command[1]].name}')

            # print('######################################')
            '''if command[0] == 7 and command[1] == 8:
                self.environment_set_action(self.factory.agvs[command[2]], self.factory.warehouses[0],
                                            self.factory.machines[0])
            elif command[0] == 9 and command[1] == 10:
                self.environment_set_action(self.factory.agvs[command[2]], self.factory.machines[0],
                                            self.factory.machines[1])
            elif command[0] == 11 and command[1] == 12:
                self.environment_set_action(self.factory.agvs[command[2]], self.factory.machines[1],
                                            self.factory.machines[2])
            elif command[0] == 13 and command[1] == 6:
                self.environment_set_action(self.factory.agvs[command[2]], self.factory.machines[2],
                                            self.factory.warehouses[0])'''

        # print(self.factory.agvs[0].coupling_master == self.factory.agvs[0])
        # print(self.factory.agvs[2].coupling_master == self.factory.agvs[1])

    #def environment_set_action(agv, loading, unloading):
    #    if agv.is_free:
    #        product = unloading.input_products[0]
    #        if loading.has_product(product):
    #            agv.deliver(loading, unloading, product)

    def environment_set_action(self, agv, output_object, input_object):
        if self.coupling_command:
            self._couple(agv, output_object)
        else:
            self._deliver(agv, input_object, output_object)

    def _deliver(self, agv, input_object, output_object):
        if not agv.is_free:
        # if agv.loaded_product is not None:
            return
        for product in input_object.input_products:
            if output_object is not None:
                self.agv_positioning = self.factory.get_agv_needed_for_product(product, agv)
                if self.agv_positioning[0] > 1 or self.agv_positioning[1] > 1:
                    self.coupling_command = True
                    self.coupling_master = agv
                    self.agv_couple_count = self.agv_positioning[0] * self.agv_positioning[1] - 1
                    agv.coupling(self.coupling_master, [0, 0], self.agv_couple_count, output_object, input_object,
                                 product, self.agv_positioning)
                else:
                    agv.deliver(output_object, input_object, product)

    def _couple_simple(self, agv, output_object):
        if self.agv_couple_count > 0:
            for agv in self.factory.agvs:
                if agv.is_free:
                    count = 0
                    pos = [0, 0]
                    for length in range(self.agv_positioning[1]):
                        for width in range(self.agv_positioning[0]):
                            if count == self.agv_positioning[0] * self.agv_positioning[1] - self.agv_couple_count:
                                pos = [width, length]
                                break
                            count += 1
                        if pos[0] != 0 or pos[1] != 0:
                            break
                    agv.coupling(self.coupling_master, pos, output_object=output_object)
                    self.agv_couple_count -= 1
        else:
            self.coupling_command = False

    def _couple(self, agv, output_object):
        if self.agv_couple_count > 0:
            if agv.is_free:
                count = 0
                pos = [0, 0]
                for length in range(self.agv_positioning[1]):
                    for width in range(self.agv_positioning[0]):
                        if count == self.agv_positioning[0] * self.agv_positioning[1] - self.agv_couple_count:
                            pos = [width, length]
                        count += 1
                agv.coupling(self.coupling_master, pos, output_object=output_object)
                self.agv_couple_count -= 1
                if self.agv_couple_count <= 0:
                    self.coupling_command = False
                return
        else:
            self.coupling_command = False

    @staticmethod
    def _unload_agv(agv, unloading):
        agv.unload(unloading)

    def _block_until_synchronized(self):
        all_synchronized = True
        for i in range(1000):
            all_synchronized = True
            for agv in self.factory.agvs:
                if agv.step_counter_last < self.step_counter - 1:
                    all_synchronized = False
                    # print("\n bla")
                    break
            if all_synchronized:
                break
            time.sleep(self.sleep_time)
        if not all_synchronized:
            print(self.step_counter, self.factory.agvs[0].step_counter_next,
                  self.factory.agvs[0].step_counter_last, self.factory.agvs[1].step_counter_last,
                  self.factory.agvs[2].step_counter_last, self.factory.agvs[3].step_counter_last,
                  self.factory.agvs[4].step_counter_last, self.factory.agvs[5].step_counter_last)

    def _simulate_factory_objects(self):
        for agv in self.factory.agvs:
            agv.step(self.time_step, self.step_counter)
        for warehouse in self.factory.warehouses:
            warehouse.step(self.time_step)
        for machine in self.factory.machines:
            machine.step(self.time_step)

    def render(self):
        """
        Display a 2D list of color data using pygame.
        """
        if self.render_graphics:
            color_data = self.factory.get_color_grid()

            # Main loop to draw the colors and handle events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.factory.print_status()
                if event.type == pygame.QUIT:
                    pygame.quit()
                    # sys.exit() not for including in GUI

            # Draw the color data
            for y in range(self.height):
                for x in range(self.width):
                    pygame.draw.rect(self.screen, color_data[x][y],
                                     (x * self.pixel_size, y * self.pixel_size, self.pixel_size, self.pixel_size))

            # Draw a grid
            for y in range(self.height):
                pygame.draw.line(self.screen, [0, 0, 0], [0, y * self.pixel_size + self.pixel_size],
                                 [self.width * self.pixel_size, y * self.pixel_size + self.pixel_size])
            for x in range(self.width):
                pygame.draw.line(self.screen, [0, 0, 0], [x * self.pixel_size + self.pixel_size, 0],
                                 [x * self.pixel_size + self.pixel_size, self.height * self.pixel_size])

            # Draw the AGVs
            for agv in self.factory.agvs:
                agv_pos = [agv.pos_x, agv.pos_y]
                pygame.draw.rect(self.screen, [0, 0, 0],
                                 (agv_pos[0] * self.pixel_size, agv_pos[1] * self.pixel_size,
                                  agv.width / self.reference_size * self.pixel_size, agv.length / self.reference_size * self.pixel_size),
                                 border_radius=3)
                font = pygame.font.SysFont('arial', 15)
                text = font.render(agv.name[-1], True, (255, 255, 255))
                #self.screen.blit(text, [self.pixel_size * agv.pos_x, self.pixel_size * agv.pos_y])

            for agv in self.factory.agvs:
                if agv.loaded_product is not None:
                    agv_pos = [agv.pos_x, agv.pos_y]
                    pygame.draw.rect(self.screen, [180, 180, 0],
                                     (agv_pos[0] * self.pixel_size + 2,
                                      agv_pos[1] * self.pixel_size + 2,
                                      agv.loaded_product.width / self.reference_size * self.pixel_size, agv.loaded_product.length / self.reference_size * self.pixel_size))

            # Draw machines data
            for machine in self.factory.machines:
                font = pygame.font.SysFont('arial', 12)
                text = font.render(machine.get_status() +
                                   " I:" + str(len(machine.buffer_input_load)) + "/" + str(machine.buffer_input[0]) +
                                   " O:" + str(len(machine.buffer_output_load)) + "/" + str(machine.buffer_output[0]), True, (0, 0, 0))
                self.screen.blit(text, [self.pixel_size * machine.pos_x, self.pixel_size * machine.pos_y])

                font = pygame.font.SysFont('arial', 12)
                text = font.render(str(int(machine.rest_process_time)), True, (0, 0, 0))
                self.screen.blit(text, [self.pixel_size * machine.pos_x, self.pixel_size * machine.pos_y + self.pixel_size / 2])

            for warehouse in self.factory.warehouses:
                font = pygame.font.SysFont('arial', 12)
                text = font.render(str(int(warehouse.rest_process_time)), True, (0, 0, 0))
                self.screen.blit(text, [self.pixel_size * warehouse.pos_x, self.pixel_size * warehouse.pos_y + self.pixel_size / 2])

            node_position = self.assignment_node_to_agv_warehouse_machine()
            for key, value in node_position.items():
                font = pygame.font.SysFont('arial', 12)
                text = font.render(str(key), True, (255, 255, 255))
                self.screen.blit(text, [self.pixel_size * value[0] + 2, self.pixel_size * value[1]])

            pygame.display.flip()

    def assignment_node_to_agv_warehouse_machine(self):
        dict_assignment_factory_object = dict()
        dict_assignment_node_position = dict()
        for i in range(len(self.list_of_factory_objects_input_output)):
            if isinstance(self.list_of_factory_objects_input_output[i], AGV):
                dict_assignment_factory_object[i] = self.list_of_factory_objects_input_output[i]
                dict_assignment_node_position[i] = [self.list_of_factory_objects_input_output[i].pos_x,
                                                    self.list_of_factory_objects_input_output[i].pos_y]
            else:
                if self.list_of_factory_objects_input_output[i] != self.list_of_factory_objects_input_output[i - 1]:
                    dict_assignment_factory_object[i] = self.list_of_factory_objects_input_output[i]
                    dict_assignment_node_position[i] = self.list_of_factory_objects_input_output[i].pos_input
                else:
                    dict_assignment_factory_object[i] = self.list_of_factory_objects_input_output[i]
                    dict_assignment_node_position[i] = self.list_of_factory_objects_input_output[i].pos_output
        return dict_assignment_node_position

    def get_machine_idle_process_times(self):
        return self.machine_idle_process_times

    def collect_data(self):
        status_dict_machine = { "idle": 0,
                        "blocked": 1,
                        "process": 2}
        for i in range(len(self.factory.agvs)):
            if ((self.factory.agvs[i].status != 'idle' or self.factory.agvs[i].status != 'unload_product') and self.factory.agvs[i].loaded_product is not None or
                    self.factory.agvs[i].coupling_master is not None and self.factory.agvs[i].coupling_master.loaded_product is not None):
                workload = 1  # if agv is loaded by itself or master is loaded
                self.agv_status_durations[self.factory.agvs[i].name]["loaded"] += 1
            else:
                workload = 0
                self.agv_status_durations[self.factory.agvs[i].name]["unloaded"] += 1


            workload += self.agvs_workload[i][-1] * len(self.agvs_workload[i])
            workload /= len(self.agvs_workload[i]) + 1
            self.agvs_workload[i].append(workload)

        for i in range(len(self.factory.machines)):
            machine_state = self.factory.machines[i].get_status()
            self.machine_status_durations[self.factory.machines[i].name][machine_state] += 1
            self.machine_idle_process_times[self.factory.machines[i].name][machine_state] += 1 * self.time_step

            workload = 0
            if machine_state == 'process':
                workload = 1
            workload += self.machines_workload[i][-1] * len(self.machines_workload[i])
            workload /= len(self.machines_workload[i]) + 1
            self.machines_workload[i].append(workload)
            if machine_state == 'idle':
                self.machine_idle_process_times[self.factory.machines[i].name]['process'] = 0
                self.machine_idle_process_times[self.factory.machines[i].name]['blocked'] = 0
            if machine_state == 'process':
                self.machine_idle_process_times[self.factory.machines[i].name]['idle'] = 0
                self.machine_idle_process_times[self.factory.machines[i].name]['blocked'] = 0
            if machine_state == 'blocked':
                self.machine_idle_process_times[self.factory.machines[i].name]['idle'] = 0
                self.machine_idle_process_times[self.factory.machines[i].name]['process'] = 0
            # print(f'Machine_Times: {self.machine_idle_process_times}')

        if self.step_counter % 10 == 0:
            self.machine_states_log.append([status_dict_machine[m.get_status()] for m in self.factory.machines])
            self.agv_states_log.append([a.status for a in self.factory.agvs])
            self.agv_commands_log.append([a.command for a in self.factory.agvs])
        if self.step_counter % 100 == 0 and self.write_files:
            machine_status_df = pd.DataFrame.from_dict(self.machine_status_durations)
            machine_status_df.to_csv(path_or_buf=self.project_path + "/data/Current_Factory/Sim_data/machine_workload.csv",
                             sep=";")
            machine_state_log_df = pd.DataFrame(self.machine_states_log, columns=[m.name for m in self.factory.machines])
            machine_state_log_df.to_csv(path_or_buf=self.project_path + "/data/Current_Factory/Sim_data/machine_state_logs.csv",
                                sep=";")
            agv_state_df = pd.DataFrame.from_dict(self.agv_status_durations)
            agv_state_df.to_csv(
                path_or_buf=self.project_path + "/data/Current_Factory/Sim_data/agv_workload.csv",
                sep=";")
            agv_state_log_df = pd.DataFrame(self.agv_states_log,
                                                columns=[a.name for a in self.factory.agvs])
            agv_state_log_df.to_csv(
                path_or_buf=self.project_path + "/data/Current_Factory/Sim_data/agv_state_logs.csv",
                sep=";")
            agv_commands_log_df = pd.DataFrame(self.agv_commands_log,
                                            columns=[a.name for a in self.factory.agvs])
            agv_commands_log_df.to_csv(
                path_or_buf=self.project_path + "/data/Current_Factory/Sim_data/agv_commands_logs.csv",
                sep=";")

        self.get_system_stock_data()
        self.get_transport_stock_data()

    def get_system_stock_data(self):
        count = 0
        for product in self.factory.products:
            if product.stored_in != self.factory.warehouses[0]:
                count += 1
        mean = (self.system_stock_data[-1] * len(self.system_stock_data) + count) / (len(self.system_stock_data) + 1)
        self.system_stock_data.append(mean)
        if self.step_counter % 100 == 0:
            self.average_system_stock_data_for_csv.append(mean)
            if self.write_files:
                average_system_stock_data_df = pd.DataFrame(self.average_system_stock_data_for_csv)
                average_system_stock_data_df.to_csv(
                        path_or_buf=self.project_path + "/data/Current_Factory/Sim_data/average_system_stock_over_time.csv",
                        sep=";")
        self.system_stock_data_for_csv.append(count)
        if self.write_files:
            system_stock_data_df = pd.DataFrame(self.system_stock_data_for_csv)
            system_stock_data_df.to_csv(
                path_or_buf=self.project_path + "/data/Current_Factory/Sim_data/system_stock_over_time.csv",
                sep=";")

    def get_transport_stock_data(self):
        count = 0
        for product in self.factory.products:
            for agv in self.factory.agvs:
                if product.stored_in == agv:
                    count += 1
                    break
        mean = (self.system_transport_data[-1] * len(self.system_transport_data) + count) / (len(self.system_transport_data) + 1)
        self.system_transport_data.append(mean)
        if self.step_counter % 100 == 0:
            self.average_system_transport_data_for_csv.append(mean)
            if self.write_files:
                transport_stock_data_df = pd.DataFrame(self.average_system_transport_data_for_csv)
                transport_stock_data_df.to_csv(
                    path_or_buf=self.project_path + "/data/Current_Factory/Sim_data/average_transport_stock_over_time.csv",
                    sep=";")
        self.system_transport_data_for_csv.append(count)
        if self.write_files:
            system_transport_data_df = pd.DataFrame(self.system_stock_data_for_csv)
            system_transport_data_df.to_csv(
                path_or_buf=self.project_path + "/data/Current_Factory/Sim_data/transport_stock_over_time.csv",
                sep=";")

    def plot_durations(self, show_result=False):
        # plt.figure(1)
        plt.clf()
        plt.subplot(1, 2, 1)  # row 1, column 2, count 1
        plt.plot(self.system_transport_data_for_csv, label='transport stock', color='r', linewidth=4)
        plt.plot(self.system_stock_data_for_csv, label='system stock', color='g', linewidth=2)
        if show_result:
            plt.title('Result')
        else:
            plt.title('Running...')
        plt.xlabel('Time in 0.1s Steps')
        plt.ylabel('Count')
        plt.legend()

        # using subplot function and creating plot two
        # row 1, column 2, count 2
        plt.subplot(1, 2, 2)
        for i in range(len(self.agvs_workload)):
            plt.plot(self.agvs_workload[i], label='AGV ' + str(i), linewidth=4-i)
        for i in range(len(self.machines_workload)):
            plt.plot(self.machines_workload[i], label='Machine ' + str(i), linewidth=4-i)
        plt.title('Workload')
        plt.xlabel('Time in 0.1s Steps')
        plt.ylabel('Count')
        plt.legend()

        plt.tight_layout(pad=1.0)
        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())