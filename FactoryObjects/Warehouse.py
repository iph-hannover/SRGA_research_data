import config
import numpy as np


class Warehouse:
    def __init__(self):
        self.id = config.warehouse['id']
        self.name = config.warehouse['name']
        self.length = config.warehouse['length']
        self.width = config.warehouse['width']
        self.pos_x = config.warehouse['pos_x']
        self.pos_y = config.warehouse['pos_y']
        self.pos_input = config.warehouse['pos_input']
        self.pos_output = config.warehouse['pos_output']
        np_pos_warehouse = np.array([int(self.pos_x), int(self.pos_y)])
        np_warehouse_pos_input = np.array(self.pos_input)
        np_warehouse_pos_output = np.array(self.pos_output)
        self.local_pos_input = (np_pos_warehouse + np_warehouse_pos_input).tolist()
        self.local_pos_output = (np_pos_warehouse + np_warehouse_pos_output).tolist()
        self.output_products = config.warehouse['output_products']
        self.input_products = config.warehouse['input_products']
        self.process_time = config.warehouse['process_time']
        self.rest_process_time = config.warehouse['process_time']
        self.buffer_output = config.warehouse['buffer_output']
        self.buffer_output_load = []
        self.buffer_input = config.warehouse['buffer_input']
        self.buffer_input_load = []
        self.end_product_store = []
        self.temp_store = []
        self.loading_time_output = config.warehouse['loading_time_output']
        self.loading_time_input = config.warehouse['loading_time_input']
        self.upstream_id = config.warehouse['upstream_id']
        self.downstream_id = config.warehouse['downstream_id']
        self.color = config.warehouse['base_color']
        self.factory = None
        self.status = 'idle'
        self.process_object = None
        self.list = []
        self.status = 'idle'
        self.has_new_input_product = False
        self.has_new_output_product = False

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

    def reload_settings(self):
        self.length = config.warehouse['length']
        self.width = config.warehouse['width']
        self.pos_x = config.warehouse['pos_x']
        self.pos_y = config.warehouse['pos_y']
        self.process_time = config.warehouse['process_time']

    def reset(self):
        self.end_product_store = []
        self.temp_store = []
        self.buffer_output_load = []
        self.buffer_input_load = []
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
        return None

    def get_color(self):
        return self.color
    
    def has_product(self, product_name):
        for product in self.buffer_output_load:
            if product.name == product_name:
                return True
        for product in self.temp_store:
            if product.name == product_name:
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
    
    def step(self, time):
        if self.status == 'idle':
            if self.rest_process_time == self.process_time:
                self.status = 'process'
        if self.status == 'process':
            if self.process(time):
                self.status = 'idle'

    def process(self, time):
        if self.rest_process_time > 0.0 and self.rest_process_time != self.process_time:
            self.rest_process_time -= time
            return False
        index = -1  # the implementation of multiple product handling is not done yet
        for i in range(len(self.output_products)):
            if len(self.buffer_output_load) >= self.buffer_output[i]:
                continue
            else:
                index = i
            break
        if index == -1:
            return False
        self.rest_process_time -= time
        if self.rest_process_time <= 0.0:
            handover_product = self.factory.create_product(self.output_products[index])
            handover_product.stored_in = self
            self.buffer_output_load.append(handover_product)
            self.rest_process_time = self.process_time
            self.has_new_output_product = True
            return True
        return False

    def find_output_product(self, product_name):
        for output_product_name in self.output_products:
            if product_name == output_product_name:
                return True
        return False

    def handover_input_product(self, product):
        self.has_new_input_product = True
        for product_name in self.input_products:
            if product.name == product_name:
                product.stored_in = self
                self.end_product_store.append(product)
                return True
        product.stored_in = self
        self.temp_store.append(product)
        return True

    def handover_output_product(self, product_name):
        for product in self.temp_store:
            if product.name == product_name:
                handover_product = product
                self.temp_store.remove(handover_product)
                return handover_product
        # handover_product = self.factory.create_product(product_name)
        # handover_product.stored_in = self
        # return handover_product

        if not self.find_output_product(product_name):
            return None
        if len(self.buffer_output_load) <= 0:
            return None
        product = self.buffer_output_load[0]
        self.buffer_output_load.pop(0)
        return product

    def get_middle_position(self):
        return [(self.pos_input[0] + self.pos_output[0]) / 2, (self.pos_input[1] + self.pos_output[0]) / 2]

    def get_production_rest_time_percent(self):
        return self.rest_process_time / self.process_time
