import os
import sys
import numpy as np
import torch
import scipy.io as scio
import matplotlib.pyplot as plt
import random
from utils.keyframes import *
from utils.utils import *

# 相对变换后的位置边界
crop_size = 256

# 需要的关节位置
# need_index = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
need_index = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

mat_data_dir = './data/SLR_dataset/xf500_body_depth_mat'
data = []
label = []
index = 0
start_idx = 96
end_idx = 196
keyframe_num = 36
for label_dir in os.listdir(mat_data_dir):
    if index >= start_idx and index < end_idx:
        print(label_dir)
        one_label_dir = os.path.join(mat_data_dir, label_dir)
        one_label_data = []
        for one_mat in os.listdir(one_label_dir):
            # 读取mat数据
            one_data = scio.loadmat(os.path.join(one_label_dir, one_mat))['data'].astype(np.float32)
            # 提取关键帧
            key_indexes = extract_keyframes_indexes(one_data, keyframe_num)
            # 剔除小于keyframe_num帧数据
            if len(key_indexes) < keyframe_num:
                continue
            key_frames = one_data[key_indexes]
            # save_one_mat2txt(key_frames[:, need_index], 'test_server.txt')
            one_label_data.append(key_frames)
            pass
        one_label_data_array = np.array(one_label_data, dtype=np.float32)
        data.append(one_label_data_array)
        one_label = int(index - start_idx)
        label.append(np.ones(len(one_label_data)) * one_label)
    index += 1

# 截取部分需要的关节点
data_array = np.array(data, dtype=np.float32).reshape(-1, keyframe_num, 50)[:, :, need_index]
label_array = np.array(label, dtype=np.int16).reshape(-1, 1)
# 转换为相对坐标
for i in range(len(data_array)):
    for j in range(len(data_array[i])):
        data_array[i][j] = abs2rel(data_array[i][j], crop_size)

# 打乱数据
shuffle_index = list(range(len(label_array)))
random.shuffle(shuffle_index)
data_array_shuffled = data_array[shuffle_index]
label_array_shuffled = label_array[shuffle_index]

# 保存文件
data_npy_name = "SLR_S" + str(start_idx) + "_E" + str(end_idx) + "_body_data.npy"
label_npy_name = "SLR_S" + str(start_idx) + "_E" + str(end_idx) + "_body_label.npy"
np.save(os.path.join("./data/SLR_dataset/processed", data_npy_name), data_array_shuffled, allow_pickle=True)
np.save(os.path.join("./data/SLR_dataset/processed", label_npy_name), label_array_shuffled, allow_pickle=True)
pass