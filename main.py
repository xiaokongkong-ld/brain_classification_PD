# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

DATA_PATH='/media/ncclab/database5/lidan/HCP_1200_part_0'
save_dir = DATA_PATH + '/hcp_part_0_time_series_list_RL'
load_dir = save_dir + '.npz'

test = np.load(load_dir
            , allow_pickle=True
            )['a']

print(test.shape)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
