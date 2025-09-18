import time

import config


class LoadingStation:
    def __init__(self):
        self.id = config.loading_station['id']
        self.name = config.loading_station['name']
        self.length = config.loading_station['length']
        self.width = config.loading_station['width']
        self.pos_x = config.loading_station['pos_x']
        self.pos_y = config.loading_station['pos_y']
        self.charging_time = config.loading_station['charging_time']
        self.capacity = config.loading_station['capacity']
        self.list = []
        self.last_time_in_use = time.time()
        self.in_use_by = None
        self.logout_time = 10  # seconds

    def create_list(self):
        self.list = []
        self.list.append(self.id)
        self.list.append(self.name)
        self.list.append(self.length)
        self.list.append(self.width)
        self.list.append(self.pos_x)
        self.list.append(self.pos_y)
        self.list.append(self.charging_time)
        self.list.append(self.capacity)
        return self.list

    def get_block_type(self, pos):
        return None

    def get_color(self):
        return [242, 242, 0]

    def register_agv(self, agv):
        if agv == self.in_use_by:
            self.last_time_in_use = time.time()
            return True
        if time.time() - self.last_time_in_use > self.logout_time:
            self.last_time_in_use = time.time()
            self.in_use_by = agv
            return True
        return False
