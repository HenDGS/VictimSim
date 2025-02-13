import sys
import os
import time
## importa classes
from environment import Env
from explorer import Explorer
from rescuerr import Rescuer


def main(data_folder_name):
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    # Instantiate the environment
    env = Env(data_folder)

    # config files for the agents
    rescuer_file = os.path.join(data_folder, "rescuer_config.txt")
    explorer_file = os.path.join(data_folder, "explorer_config.txt")

    # Instantiate agents rescuer and explorer
    resc = Rescuer(env, rescuer_file,1)
    resc2 = Rescuer(env, rescuer_file,2)
    resc3 = Rescuer(env, rescuer_file,3)
    resc4 = Rescuer(env, rescuer_file,4)

    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    exp = Explorer(env, explorer_file, resc, 1)
    exp2 = Explorer(env, explorer_file, resc2, 2)
    exp3 = Explorer(env, explorer_file, resc3, 3)
    exp4 = Explorer(env, explorer_file, resc4, 4)

    # Run the environment simulator
    env.run()


if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""

    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        # data_folder_name = "data"
        # data_folder_name = "datasets/data_12x12_10vic"
        # data_folder_name = "datasets/data_20x20_42vic"
        # data_folder_name = "datasets/data_100x80_132vic"
        data_folder_name = "datasets/data_100x80_225vic"

    main(data_folder_name)
