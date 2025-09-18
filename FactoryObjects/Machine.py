import config
import numpy as np


class Machine:
    def __init__(self):
        self.id = config.machine['id']
        self.name = config.machine['name']
        self.length = config.machine['length']
        self.width = config.machine['width']
        self.pos_x = config.machine['pos_x']
        self.pos_y = config.machine['pos_y']
        self.pos_input = config.machine['pos_input']
        self.pos_output = config.machine['pos_output']
        np_pos_machine = np.array([int(self.pos_x), int(self.pos_y)])
        np_machine_pos_input = np.array(self.pos_input)
        np_machine_pos_output = np.array(self.pos_output)
        self.local_pos_input = (np_pos_machine + np_machine_pos_input).tolist()
        self.local_pos_output = (np_pos_machine + np_machine_pos_output).tolist()
        self.input_products = config.machine['input_products']
        self.output_products = config.machine['output_products']
        self.process_time = config.machine['process_time']
        self.rest_process_time = config.machine['process_time']
        self.buffer_input = config.machine['buffer_input']
        self.buffer_output = config.machine['buffer_output']
        self.buffer_input_load = []
        self.buffer_output_load = []
        self.loading_time_input = config.machine['loading_time_input']
        self.loading_time_output = config.machine['loading_time_output']
        self.color = config.machine['base_color']
        self.factory = None
        self.status = 'idle'
        self.process_object = None
        self.list = []
        self.has_new_input_product = False
        self.has_new_output_product = False

        self.amount_of_resulting_products = 1

    def create_list(self):
        self.list = []
        self.list.append(self.id)
        self.list.append(self.name)
        self.list.append(self.length)
        self.list.append(self.width)
        self.list.append(self.pos_x)
        self.list.append(self.pos_y)
        self.list.append(self.pos_input)
        self.list.append(self.pos_output)
        self.list.append(self.input_products)
        self.list.append(self.output_products)
        self.list.append(self.process_time)
        self.list.append(self.buffer_input)
        self.list.append(self.buffer_output)
        self.list.append(self.loading_time_input)
        self.list.append(self.loading_time_output)
        return self.list

    def print_machine(self):
        print(f'Inside CLASS Machine - Machine ID = {self.id}')
        print(f'Inside CLASS Machine - Machine Name = {self.name}')
        print(f'Inside CLASS Machine - Machine Length = {self.length}')
        print(f'Inside CLASS Machine - Machine Width = {self.width}')
        print(f'Inside CLASS Machine - Machine X-POS = {self.pos_x}')
        print(f'Inside CLASS Machine - Machine Y-POS = {self.pos_y}')
        print(f'Inside CLASS Machine - Machine Process Time = {self.process_time}')
        print(f'Inside CLASS Machine - Machine Input Position = {self.pos_input}')
        print(f'Inside CLASS Machine - Machine Output Position = {self.pos_output}')
        print(f'Inside CLASS Machine - Machine Local Input Position = {self.local_pos_input}')
        print(f'Inside CLASS Machine - Machine Local Output Position = {self.local_pos_output}')
        print(f'Inside CLASS Machine - Machine Input Products = {self.input_products}')
        print(f'Inside CLASS Machine - Machine Output Products = {self.output_products}')
        print(f'Inside CLASS Machine - Machine Input Buffer Size = {self.buffer_input}')
        print(f'Inside CLASS Machine - Machine Output Buffer Size = {self.buffer_output}')
        print(f'Inside CLASS Machine - Machine Input Loading Time = {self.loading_time_input}')
        print(f'Inside CLASS Machine - Machine Output Loading Time = {self.loading_time_output}')

    def reload_settings(self):
        self.length = config.machine['length']
        self.width = config.machine['width']
        self.pos_x = config.machine['pos_x']
        self.pos_y = config.machine['pos_y']
        self.process_time = config.machine['process_time']

    def reset(self):
        self.buffer_input_load = []
        self.buffer_output_load = []
        self.status = 'idle'
        self.process_object = None
        self.rest_process_time = self.process_time

    def get_type(self):
        return self.name + " " + self.id

    def get_block_type(self, pos):
        if pos == self.pos_input:
            if pos == self.pos_output:
                return "input_output"
            return "input"
        elif pos == self.pos_output:
            if pos == self.pos_input:
                return "input_output"
            return "output"
        return "machine_block"

    def get_color(self):
        return self.color

    def has_product(self, product_name):
        if product_name in self.output_products and len(self.buffer_output_load) > 0:
            return True
        return False

    def get_has_new_input_product(self):
        if self.has_new_input_product:
            self.has_new_input_product = False
            return True
        return False

    def get_has_new_output_product(self):
        if self.has_new_output_product:
            self.has_new_output_product = False
            return True
        return False

    def handover_input_product(self, product):
        for i in range(len(self.input_products)):
            if product.name != self.input_products[i]:
                continue
            if len(self.buffer_input_load) >= self.buffer_input[i]:
                continue
            product.stored_in = self
            self.buffer_input_load.append(product)
            self.has_new_input_product = True
            return True
        return False

    def handover_output_product(self, product_name):
        if not self.find_output_product(product_name):
            return None
        if len(self.buffer_output_load) <= 0:
            return None
        product = self.buffer_output_load[0]
        self.buffer_output_load.pop(0)
        if len(self.buffer_output_load) > 0:
            self.has_new_output_product = True
        return product

    def find_output_product(self, product_name):
        for output_product_name in self.output_products:
            if product_name == output_product_name:
                return True
        return False

    def get_buffer_status(self):
        input_priority = 0  # 0 ... 4 None Low, Mid, High Critical
        output_priority = 0
        if self.buffer_input[0] * 0.5 <= len(self.buffer_input_load) < self.buffer_input[0]:
            # If input_buffer is 50 to 99 % full
            input_priority = 1
        elif 0 < len(self.buffer_input_load) < self.buffer_input[0] * 0.5:
            # If input_buffer is 0 to 50 % full
            input_priority = 2
        elif len(self.buffer_input_load) == 0:
            # If input_buffer is empty
            input_priority = 3
            if self.rest_process_time == self.process_time:
                # If input_buffer is empty and no process is running
                input_priority = 4

        if 0 < len(self.buffer_output_load) <= self.buffer_output[0] * 0.5:
            # If output_buffer is 1 to 50 % full
            output_priority = 1
        if self.buffer_output[0] * 0.5 < len(self.buffer_output_load) < self.buffer_output[0]:
            # If output_buffer is 50 to 99 % full
            output_priority = 2
        elif len(self.buffer_output_load) == self.buffer_output[0]:
            # If output_buffer is full (100 %)
            output_priority = 3
            if self.rest_process_time == 0.0:
                # If output_buffer is full (100 %) and no process is running
                output_priority = 4
        return input_priority, output_priority

    def get_status(self):
        if self.status == 'process' and self.rest_process_time == 0:
            return 'stopped'
        return self.status

    def step(self, time):
        if self.status == 'idle':
            if self.rest_process_time == self.process_time and len(self.buffer_input_load) > 0:
                self.status = 'process'
        if self.status == 'process':
            if self.process(time):
                self.status = 'idle'
        if self.status == 'blocked':
            if self.unload_process_buffer():
                self.status = 'idle'

    def process(self, time):
        if self.rest_process_time == self.process_time:
            self.process_object = self.buffer_input_load[0]
            self.buffer_input_load.pop(0)
            self.factory.change_product(self.process_object, self.output_products[0])

        self.rest_process_time -= time
        if self.rest_process_time <= 0.0:
            return self.unload_process_buffer()
        return False

    def unload_process_buffer(self):
        if len(self.buffer_output_load) + self.amount_of_resulting_products <= self.buffer_output[0]:
            self.rest_process_time = self.process_time
            self.buffer_output_load.append(self.process_object)
            if self.amount_of_resulting_products > 1:
                for i in range(self.amount_of_resulting_products - 1):
                    new_product = self.factory.create_product(self.process_object.name)
                    self.buffer_output_load.append(new_product)
            self.process_object = None
            self.has_new_output_product = True
            return True
        else:
            self.rest_process_time = 0.0
            self.status = 'blocked'
        return False

    def get_middle_position(self):
        return [(self.pos_input[0] + self.pos_output[0]) / 2, (self.pos_input[1] + self.pos_output[0]) / 2]

    def get_production_rest_time_percent(self):
        return self.rest_process_time / self.process_time
