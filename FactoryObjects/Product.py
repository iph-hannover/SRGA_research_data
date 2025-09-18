import config


class Product:
    def __init__(self):
        self.id = config.product['id']
        self.name = config.product['name']
        self.length = config.product['length']
        self.width = config.product['width']
        self.weight = config.product['weight']
        self.stored_in = None

