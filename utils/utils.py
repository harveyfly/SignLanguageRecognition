import matplotlib.pyplot as plt
import numpy as np

def plot_one_data(data):
    '''plot one init data'''
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.scatter(data[0::2], data[1::2])
    plt.show()

# 绝对坐标转化为相对坐标
def abs2rel(data, crop_size):
    data_x = data[0::2]
    data_y = data[1::2]
    x_min = np.min(data_x)
    x_max = np.max(data_x)
    y_min = np.min(data_y)
    y_max = np.max(data_y)
    data[0::2] = (data_x - x_min) / (x_max - x_min) * crop_size
    data[1::2] = (data_y - y_min) / (y_max - y_min) * crop_size
    return data

def read_dict_table(path):
    data = dict()
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            line_list = line.split()
            key = int(line_list[0])
            value = str(line_list[1])
            data[key] = value
    return data

def class_index2name(dict_table, index, start_index=0):
    if start_index + index >= 500:
        return None
    return dict_table[start_index + index]

def save_one_mat2txt(one_mat, txt_path):
    np.savetxt(txt_path, one_mat, fmt='%.1f', delimiter=',', newline=',')

    