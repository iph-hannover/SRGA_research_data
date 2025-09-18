project_name = 'Zell FTF'

factory = dict(
    name='default_factory',  # init name of the factory
    length=20,  # m | length of the factory
    width=10,  # m | width of the factory
    cell_size=1.0  # m | size of a factory cell
)

machine = dict(
    id=0,  # init id of machine
    name='default_machine',  # init name of machine
    length=2,  # m | length of the machine
    width=1,  # m | width of the machine
    pos_x=0,  # x_position of machine in factory grid
    pos_y=0,  # y_position of machine in factory grid
    pos_input=[0, 0],  # x-y_position of input (drain) in factory grid
    pos_output=[1, 0],  # x-y_position of output (source) in factory grid
    input_products=['default_product_1'],  # names of the input products of the machine (drain)
    output_products=['default_product_1'],  # names of the output products of the machine (source)
    process_time=30,  # s | processing time, the first entry is for the first product etc...
    buffer_input=[1],  # pcs | buffer size of the input, the first entry is for the first product etc...
    buffer_output=[1],  # pcs | buffer size of the output, the first entry is for the first product etc...
    loading_time_input=[5],  # s | loading time of the input, the first entry is for the first product etc...
    loading_time_output=[5],  # s | loading time of the output, the first entry is for the first product etc...
    upstream_id=['W0'],  # id of the upstream machine/warehouse, the first entry is for the first product etc...
    downstream_id=['W0'],  # id of the downstream machine/warehouse, the first entry is for the first product etc...
    base_color=[72, 96, 96]  # Color of the main block [100, 100, 255]
)

warehouse = dict(
    id='W' + str(0),  # id of the warehouse
    name='default_warehouse',  # name of the warehouse
    length=2,  # m | length of the warehouse
    width=1,  # m | width of the warehouse
    pos_x=0,  # x_position of warehouse in factory grid
    pos_y=0,  # y-position of warehouse in factory grid
    pos_input=[0, 0],  # x-y_position of drain (input) in warehouse grid
    pos_output=[1, 0],  # x-y_position of source (output) in warehouse grid
    output_products=['default_product_1'],
    input_products=['default_product_1'],
    process_time=10,  # s | time, in which interval products are made available, '-1' means parts are constantly
    #                     provided, the first entry is for the first product etc...
    buffer_output=[1],  # pcs | number of products which are provided per product, '-1' means an infinite number
    buffer_input=[-1],  # pcs | number of products, which can be stored in the warehouse '-1' means an infinite number
    loading_time_output=[10],  # s | loading time of the source (output), the first entry is for the first product etc...
    loading_time_input=[10],  # s | loading time of the drain (input), the first entry is for the first product etc...
    upstream_id=['M0'],  # id of the upstream machine/warehouse, the first entry is for the first product etc...
    downstream_id=['M0'],  # id of the downstream machine/warehouse, the first entry is for the first product etc...
    base_color=[168, 148, 148]  # Color of the main block [255, 150, 100]
)

loading_station=dict(
    id='LS' + str(0),  # id of the loading station
    name='default_loading_station',  # name of the loading station
    length=1,  # m | length of the loading station
    width=1,  # m | width of the loading station
    pos_x=0,  # x-position of the loading station in the factory grid
    pos_y=0,  # y-position of the loading station in the factory grid
    charging_time=1,  # %/min | loading time in % per minute
    capacity=1  # numbers of AGVs, which can be loaded at once, '-1' means an infinite number
)

product_type = dict(
    name='default_product_type', # name of the product type
    length=500,  # mm | length of the product
    width=500,  # mm | width of the product
    weight=50  # kg | weight of the product
)

product = dict(
    id='P' + str(0),  # id of the product
    name='default_product',  # name of the product
    length=500,  # mm | length of the product
    width=500,  # mm | width of the product
    weight=50  # kg | weight of the product
)

agv = dict(
    name="0",
    max_speed=1,  # m/s
    load_speed=0.95,  # m/s
    couple_speed=0.95,  # m/s
    length=500,  # mm
    width=500,  # mm
    payload=500,  # kg
    base_color=[64, 64, 64]  # Color of the AGV in rgb
)

forklift = dict(
    id='FL' + str(0),  # init id of forklift
    max_speed=1,  # m/s
    length=1000,  # mm
    width=1000,  # mm
    payload=1500  # kg
)
