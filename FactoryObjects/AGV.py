import math
import time
import config
import threading


class AGV:
    def __init__(self, start_position=None, static=False):
        if start_position is None:
            self.start_position = [0, 0]
        else:
            self.start_position = start_position
        self.name = config.agv['name']
        self.max_speed = config.agv['max_speed']
        self.load_speed = config.agv['load_speed']
        self.couple_speed = config.agv['couple_speed']
        self.length = config.agv['length']
        self.width = config.agv['width']
        self.payload = config.agv['payload']
        self.pos_x = self.start_position[0]
        self.pos_y = self.start_position[1]
        self.color = config.agv['base_color']
        self.factory = None
        self.is_free = True  # True when no product is loaded
        self.command = 'idle'  # there are 3 commands: 'idle', 'move', 'deliver'
        self.status = 'idle'  # there are 5 statuses: 'idle', 'move_to_input', 'move_to_output', 'load_product', 'unload_product'
        self.idle_time = 0
        self.max_idle_time = 6000  # seconds
        self.move_target = [self.pos_x, self.pos_y]
        self.target_queue = [[self.pos_x, self.pos_y]]
        self.output_object = None
        self.input_object = None
        self.loaded_product = None
        self.target_product = None
        self.coupling_master = None
        self.agv_couple_count = 1
        self.coupling_time_max = 2.0
        self.coupling_time = 0.0
        self.coupling_position = [3, 3]
        self.coupling_formation_position = [0, 0]
        self.static = static

        # Threading !!! There is a possibility of an asynchronous behavior. If this is happening implement frame counter
        self.thread_running = True
        self.thread_waiting = True
        self.time_step = 0.0
        self.waited_time = 0.0
        self.sleep_time = 0.0001
        if static == False:
            self.run_thread = threading.Thread(target=self.run)
            self.run_thread.start()
        self.step_counter_last = 0
        self.step_counter_next = 0
        self.coupled_size = [1, 1]
        self.min_distance = 0.2  # standard = 0.1

    def reload_settings(self):
        self.max_speed = config.agv['max_speed']
        self.length = config.agv['length']
        self.width = config.agv['width']
        self.payload = config.agv['payload']
        self.pos_x = self.start_position[0]
        self.pos_y = self.start_position[1]

    def reset(self):
        self.is_free = True
        self.command = 'idle'
        self.status = 'idle'
        self.output_object = None
        self.input_object = None
        self.loaded_product = None
        self.target_product = None
        self.pos_x = self.start_position[0]
        self.pos_y = self.start_position[1]

    def set_target_and_move(self, target):
        if self.factory.use_paths:
            self.target_queue = self.factory.get_path_queue((round(self.pos_x), round(self.pos_y)), target)
            self.move_target = self.target_queue[0]
            # print(f"SETTARGET-MOVE: AGV: {self.name} to Target{target}, QUEUE: {self.target_queue}")

        else:
            self.move_target = target
        self.command = 'move'

    def set_target(self, target):
        if self.factory.use_paths:
            self.target_queue = self.factory.get_path_queue((round(self.pos_x), round(self.pos_y)), target)
            self.move_target = self.target_queue[0]
            # print(f"SETTARGET: AGV: {self.name} to Target{target}, QUEUE: {self.target_queue}")
        else:
            self.move_target = target

    def unload(self, input_object):
        if self.loaded_product is None:
            self.command = 'idle'
            self.status = 'idle'
            self.is_free = True
            self.move_to_loading_station()
            return
        self.input_object = input_object
        self.set_target(input_object.pos_input)
        # self.move_target = input_object.pos_input
        self.is_free = False
        self.command = 'deliver'
        self.status = 'move_to_input'

    def deliver(self, output_object, input_object, product_name):
        self.output_object = output_object
        self.input_object = input_object
        self.target_product = product_name
        self.is_free = False
        self.command = 'deliver'

    def coupling(self, coupling_master, paring_formation_position, agv_couple_count=1, output_object=None,
                 input_object=None, product_name=None, coupled_size=None):
        self.is_free = False
        if coupled_size is None:
            coupled_size = [1, 1]
        self.output_object = output_object
        self.input_object = input_object
        self.target_product = product_name
        self.command = 'coupling'
        self.coupling_master = coupling_master
        self.coupling_formation_position = paring_formation_position
        self.agv_couple_count = agv_couple_count
        self.coupled_size = coupled_size

    def run(self):
        while self.thread_running:
            time.sleep(self.sleep_time)
            if not self.thread_waiting:
                self.step_command()
                self.thread_waiting = True
                self.step_counter_last = self.step_counter_next

    def step(self, time_step, step_counter):
        self.step_counter_next = step_counter
        self.time_step = time_step
        self.thread_waiting = False

    def step_command(self):
        if self.command == 'move':
            if self.move_state():
                self.command = 'idle'
        elif self.command == 'deliver':
            self.deliver_state()
        elif self.command == 'coupling':
            self.coupling_agv()
        elif self.command == 'follow_master':
            self.follow_master()
        if self.command == 'idle':
            self.idle_time += self.time_step
        else:
            self.idle_time = 0
        if self.idle_time > self.max_idle_time and self.is_free:
            self.move_to_loading_station()

    def move_state(self):
        # if self.factory.use_paths:
        #     self.move_target = self.target_queue[0]
        move_vector = [self.move_target[0] - self.pos_x, self.move_target[1] - self.pos_y]
        distance = math.sqrt(math.pow(move_vector[0], 2) + math.pow(move_vector[1], 2))
        if distance < self.min_distance:
            self.pos_x = self.move_target[0]
            self.pos_y = self.move_target[1]

            if self.factory.use_paths and self.command != 'follow_master':
                if len(self.target_queue) > 1:
                    self.target_queue.pop(0)
                    self.move_target = self.target_queue[0]
                    move_vector = [self.move_target[0] - self.pos_x, self.move_target[1] - self.pos_y]
                    distance = math.sqrt(math.pow(move_vector[0], 2) + math.pow(move_vector[1], 2))
                else:
                    return True

            else:
                return True
        norm = 1 / distance
        move_vector = [norm * move_vector[0], norm * move_vector[1]]
        if self.coupling_master == self:
            speed = self.couple_speed
        elif self.loaded_product is not None:
            speed = self.load_speed
        else:
            speed = self.max_speed
        distance_vector = [move_vector[0] * speed * self.time_step, move_vector[1] * speed * self.time_step]
        self.pos_x += distance_vector[0]
        self.pos_y += distance_vector[1]
        return False

    def deliver_state(self):
        if self.status == 'idle':
            self.set_target(self.output_object.pos_output)
            # self.move_target = self.output_object.pos_output
            self.move_state()
            self.status = 'move_to_output'
        elif self.status == 'move_to_output':
            if self.move_state():
                self.status = 'load_product'
        elif self.status == 'load_product':
            product = self.output_object.handover_output_product(self.target_product)
            if self.load_product(product):
                self.set_target(self.input_object.pos_input)
                # self.move_target = self.input_object.pos_input
                self.status = 'move_to_input'
            else:
                self.is_free = True
                self.decouple()
                self.command = 'idle'
                self.status = 'idle'
                self.move_to_loading_station()
        elif self.status == 'move_to_input':
            if self.move_state():
                self.status = 'unload_product'
                self.waited_time = 0.0
        elif self.status == 'unload_product':
            if self.input_object.handover_input_product(self.loaded_product):
                self.loaded_product = None
                self.decouple()
                self.is_free = True
                self.command = 'idle'
                self.status = 'idle'
            else:
                self.unload_if_stuck()

    def load_product(self, product):
        if self.loaded_product is None and product is not None:
            if (product.width > self.width * self.coupled_size[0]
                    or product.length > self.length * self.coupled_size[1]
                    or product.weight > self.payload * self.agv_couple_count):
                return False
            self.loaded_product = product
            self.loaded_product.stored_in = self
            return True
        return False

    def move_to_loading_station(self):
        for loading_station in self.factory.loading_stations:
            if loading_station.register_agv(self):
                self.set_target_and_move([loading_station.pos_x, loading_station.pos_y])
                print(f"AGV-{self.name} get Back to loading station {(loading_station.pos_x, loading_station.pos_y)}")
                return

    def unload_if_stuck(self):
        self.waited_time += self.time_step
        if self.waited_time > self.input_object.process_time:
            self.input_object = self.factory.warehouses[0]
            self.set_target(self.input_object.pos_input)
            self.move_target = self.input_object.pos_input
            self.status = 'move_to_input'

    def coupling_agv(self):
        if self.status == 'idle':
            self.set_target(self.output_object.pos_output)
            if self.coupling_formation_position[0] != 0 and self.coupling_formation_position[1] != 0:
                self.target_queue.append((self.output_object.pos_output[0] + self.coupling_formation_position[0],
                                    self.output_object.pos_output[1] + self.coupling_formation_position[1]))
            # self.move_target = [self.output_object.pos_output[0] + self.coupling_formation_position[0],
            #                     self.output_object.pos_output[1] + self.coupling_formation_position[1]]
            #self.move_target = [self.coupling_position[0] + self.coupling_formation_position[0],
            #                    self.coupling_position[1] + self.coupling_formation_position[1]]
            self.move_state()
            self.status = 'move_to_coupling_position'
        elif self.status == 'move_to_coupling_position':
            if self.move_state():
                self.status = 'master_slave_decision'
        elif self.status == 'master_slave_decision':
            if self.coupling_master == self:
                self.status = 'wait_for_coupling'
            else:
                self.command = 'follow_master'
                self.status = 'idle'
        elif self.status == 'wait_for_coupling':
            if self.is_coupling_complete():
                self.command = 'deliver'
                self.status = 'idle'

    def is_coupling_complete(self):
        slave_count = 0
        for agv in self.factory.agvs:
            if agv != self and agv.coupling_master == self:
                if agv.command == 'follow_master':
                    slave_count += 1
        if self.agv_couple_count == slave_count:
            self.coupling_time += self.time_step
            if self.coupling_time >= self.coupling_time_max:
                self.coupling_time = 0
                return True
        return False

    def decouple(self):
        if self.coupling_master is not None:
            for agv in self.factory.agvs:
                if agv != self and agv.coupling_master == self:
                    agv.is_slave = False
                    agv.command = 'idle'
                    agv.is_free = True
                    agv.coupling_master = None
            self.coupling_master = None
            self.coupled_size = [1, 1]
            self.agv_couple_count = 1

    def follow_master(self):
        if self.coupling_master is not None:
            for _ in range(10):
                time.sleep(self.sleep_time)
                if self.coupling_master is None or self.coupling_master.step_counter_last == self.step_counter_next:
                    break
            if self.coupling_master is None:
                return
            self.move_target = [self.coupling_master.pos_x + self.coupling_formation_position[0] * self.width / (
                    self.factory.cell_size * 1000),
                                self.coupling_master.pos_y + self.coupling_formation_position[1] * self.length / (
                                        self.factory.cell_size * 1000)]
            self.move_state()

    def get_middle_position(self):
        return [self.pos_x, self.pos_y]

    def get_info(self) -> dict:
        info_object = {
            "name": self.name,
            "command": self.command,
            "status": self.status,
            "is_free": self.is_free,
            "position": self.get_middle_position(),
            "target": self.move_target,
            "queue": self.target_queue,
            "loaded_product": self.loaded_product,
            "following_master": True if self.coupling_master is not None else False,
            "coupling_Master": self.coupling_master.name if self.coupling_master is not None else None,
        }
        return info_object
