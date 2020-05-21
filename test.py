import os
import sys
import time
import datetime
import argparse

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision

# 切换当前工作目录
# os.chdir('/content/drive/My Drive/SignLanguageRecognition')

# import 子模块
from nnet.blstm import blstm
from utils.logger import *
from utils.parse_config import *
from utils.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="config/Net.cfg", help="path to model definition file")
    parser.add_argument("--model_name", type=str, default="blstm", help="used model name (lstm, blstm)")
    parser.add_argument("--data_config", type=str, default="config/SLR_dataset.cfg", help="path to data config file")
    parser.add_argument("--crop_size", type=int, default=256, help="size of each crop image")
    opt = parser.parse_args()
    print(opt)

    # 读取配置文件
    data_config = parse_data_config(opt.data_config)
    model_config = parse_model_config(opt.model_config)[opt.model_name]

    # 记录日志
    logger = Logger(data_config["log_path"])

    # 设置GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    torch.manual_seed(int(model_config["SEED"]))

    # 读取数据，转换为Tensor
    dataset_dir = data_config["dataset_dir"]
    np_data_x = np.load(os.path.join(dataset_dir, data_config['data_file_name']), allow_pickle=True)
    np_data_y = np.load(os.path.join(dataset_dir, data_config['label_file_name']), allow_pickle=True)
    data_x = torch.from_numpy(np_data_x)
    data_y = torch.from_numpy(np_data_y)
    
    # 数据集
    data_len = len(data_x)
    test_data_num = int(data_len * float(data_config['test_data_size']))
    test_x = data_x[data_len - test_data_num:]
    test_y = data_y[data_len - test_data_num:]

    logger.logger.info("Test size: " + str(test_data_num))

    # 处理模型参数
    batch_size = int(model_config["BATCH_SIZE"])
    cpu_nums = int(model_config["CPU_NUMS"])
    time_step = int(model_config["TIME_STEP"])
    input_size = int(model_config["INPUT_SIZE"])
    output_size = int(model_config["OUTPUT_SIZE"])

    # 保存的模型名称
    model_save_name = opt.model_name + "_output" + str(output_size) + "_input" + str(time_step) + "x" + str(input_size) + ".model"
    # 判断模型文件是否存在
    model_save_dir = data_config["model_save_dir"]
    model_save_path = os.path.join(model_save_dir, model_save_name)
    if not os.path.exists(model_save_path):
        logger.logger.error("model file is not existed!")
        exit()

    data_test = list(test_x.numpy().reshape(1,-1, time_step, input_size))
    data_test.append(list(test_y.numpy().reshape(-1, 1)))

    # 最外层是list，次外层是tuple，内层都是ndarray
    data_test = list(zip(*data_test))

    # 创建DataLoader
    test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=cpu_nums, pin_memory=True, shuffle=False)

    loss_func = nn.CrossEntropyLoss()
    # 测试
    best_model = torch.load(os.path.join(model_save_dir, model_save_name)).get('model').cuda()
    best_model.eval()
    final_predict = []
    ground_truth = []

    for step, (b_x, b_y) in enumerate(test_loader):
        b_x = b_x.type(torch.FloatTensor).to(device) 
        b_y = b_y.type(torch.long).to(device) 
        with torch.no_grad():
            prediction = best_model(b_x)    # rnn output
        # h_s = h_s.data        # repack the hidden state, break the connection from last iteration
        # h_c = h_c.data        # repack the hidden state, break the connection from last iteration
        loss = loss_func(prediction[:, -1, :], b_y.view(b_y.size()[0]))
        
        ground_truth = ground_truth + b_y.view(b_y.size()[0]).cpu().numpy().tolist()
        final_predict = final_predict + torch.max(prediction[:, -1, :], 1)[1].cpu().data.numpy().tolist()

    ground_truth = np.asarray(ground_truth)
    final_predict = np.asarray(final_predict)

    accuracy = float((ground_truth == final_predict).astype(int).sum()) / float(final_predict.size)
    logger.logger.info("test accuracy: " + str(accuracy))



