from VRP_Simulation.Factory_Simulation import Factory_Simulation
import time
import sys
from VRP_Simulation.dynamic_VRP import VRP_cellAGV
from Genetic_Router import GeneticRouter
import keyboard


def start_simulation_1(factory=None):
    """
    This is an event triggerd simulation.
    Each time a relevant change occurs in the factory (1. new product pops out, 2. AGV unloads product), the routing
    is recalculated.
    :return:
    """
    USE_PATHS = True

    print(sys.path[1])
    if factory is None:
        print('Factory is None')
        factory_simulation = Factory_Simulation(render=True)
    else:
        print('Factory is given')
        factory_simulation = Factory_Simulation(factory=factory, render=True)

    # vrp_router = VRP_cellAGV(factory_simulation.factory, factory_simulation, use_paths=USE_PATHS, write_data=True, with_prints=False)
    # genetic_router = GeneticRouter(factory=factory_simulation.factory, use_paths=False, write_data=True, with_prints=False)
    factory_simulation.factory.use_paths = USE_PATHS
    simulation_speed = 10
    step_counter = 0
    has_changed_counter = 0
    real_time_lapsed = 0
    start_time = time.time()
    last_time = start_time
    has_changed = True
    running = True
    while running:
        speed_time_step = factory_simulation.time_step / simulation_speed  # Calculation for the accelerated step time
        speed_sleep_time = speed_time_step / 4  # Sleep time. Small enough to not get over the time edge
        while step_counter < 60000:  # Simulation time in seconds. Real Time!
            if keyboard.is_pressed('a'):
                print('PAUSED')
                input('CONTINUE')
            time.sleep(speed_sleep_time)
            current_time = time.time()
            delta_sim_time = (current_time - last_time)
            if delta_sim_time < speed_time_step:  # if the time is not done, go back th the while commandline
                continue
            last_time += speed_time_step
            real_time_lapsed = current_time - start_time
            routing = None
            if has_changed:
                print('\n#################################################')
                # vrp_router = VRP_cellAGV(factory_simulation.factory, factory_simulation, use_paths=USE_PATHS, write_data=True, with_prints=True)
                # routing = vrp_router.get_dynamic_routing(has_changed_counter)
                genetic_router = GeneticRouter(factory=factory_simulation.factory, population_size=10, generations=10,
                                               swap_percentage_crossover=0.5, mutation_rate=0.5,
                                               fitness_rate = 0.5, fitness_weight = 1,
                                               use_paths=USE_PATHS, write_data=True, with_prints=False)
                routing = genetic_router.get_routing()
                print(f'Step-Counter = {step_counter}')
                print(f'Has_Changed-Counter = {has_changed_counter}')
                print(f'Produkte: {factory_simulation.factory.products}')
                for product in factory_simulation.factory.products:
                    print(f'    ID: {product.id}')
                    print(f'    Name: {product.name}')
                    print(f'    Stored In: {product.stored_in.name}')

                print(f'Routing = {routing}')
                # input('Press something\n')
                has_changed_counter += 1
            has_changed = factory_simulation.step(routing)
            if (step_counter % simulation_speed) == 0:
                factory_simulation.render()
                time_convert(real_time_lapsed)  # prints current time in console
            step_counter += 1


def start_simulation_2():
    print(sys.path[1])
    factory_simulation = Factory_Simulation(render=False)
    vrp_router = VRP_cellAGV(factory_simulation.factory, factory_simulation, use_paths=True)
    has_changed_counter = 0
    has_changed = True
    start_time = time.time()
    for step_counter in range(30000):
        routing = None
        if has_changed:
            vrp_router = VRP_cellAGV(factory_simulation.factory, factory_simulation, use_paths=True, write_data=False, with_prints=False)
            routing = vrp_router.get_dynamic_routing(has_changed_counter)
            print(f'Routing = {routing}')
            print(f'Step-Counter = {step_counter}')
            print(f'Has_Changed-Counter = {has_changed_counter}')
            # input('Press something\n')
            has_changed_counter += 1
        has_changed = factory_simulation.step(routing)
    end_time = time.time()
    print(f"Dauer: {end_time-start_time}")


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60

    sys.stdout.write("\rTime Lapsed = {0}:{1}:{2}".format(int(hours), int(mins), sec))


if __name__ == '__main__':
    start_simulation_1()
