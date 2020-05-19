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
os.chdir('/content/drive/My Drive/SignLanguageRecognition')

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
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    opt = parser.parse_args()
    print(opt)

    # 读取配置文件
    data_config = parse_data_config(opt.data_config)
    model_config = parse_model_config(opt.model_config)[opt.model_name]

    # 记录日志
    logger = Logger(data_config["log_path"])
    logger.logger.info(model_config)
    logger.logger.info(data_config)

    # 判断数据集是否存在
    dataset_dir = data_config["dataset_dir"]
    model_save_dir = data_config["model_save_dir"]
    if not os.path.exists(dataset_dir):
        logger.logger.info("dataset dir is not existed!")
        exit()
    os.makedirs(model_save_dir, exist_ok=True)

    # 设置GPU
    cuda_or_cpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.logger.info("device: " + cuda_or_cpu)
    device = torch.device(cuda_or_cpu)

    # 设置随机种子
    torch.manual_seed(int(model_config["SEED"]))

    # 读取数据，转换为Tensor
    np_data_x = np.load(os.path.join(dataset_dir, data_config['data_file_name']), allow_pickle=True)
    np_data_y = np.load(os.path.join(dataset_dir, data_config['label_file_name']), allow_pickle=True)
    data_x = torch.from_numpy(np_data_x)
    data_y = torch.from_numpy(np_data_y)
    # 记录数据集大小
    logger.logger.info(str(data_x.size()))
    # 打印一帧数据
    plot_one_data(data_x[0][0])


    # 分割数据集
    data_len = len(data_x)
    train_data_num = int(data_len * float(data_config['train_data_size']))
    valid_data_num = int(data_len * float(data_config['valid_data_size']))
    train_x = data_x[:train_data_num]
    train_y = data_y[:train_data_num]
    valid_x = data_x[train_data_num : train_data_num + valid_data_num]
    valid_y = data_y[train_data_num : train_data_num + valid_data_num]

    logger.logger.info("train size: " + str(train_data_num))
    logger.logger.info("valid size: " + str(valid_data_num))

    # 处理模型参数
    time_step = int(model_config["TIME_STEP"])
    input_size = int(model_config["INPUT_SIZE"])
    hidden_size = int(model_config["HIDDEN_SIZE"])
    output_size = int(model_config["OUTPUT_SIZE"])
    batch_size = int(model_config["BATCH_SIZE"])
    epoch = int(model_config["EPOCH"])
    lr = float(model_config["LEARNING_RATE"])
    drop_rate = float(model_config["DROP_RATE"])
    layers = int(model_config["LAYERS"])
    cpu_nums = int(model_config["CPU_NUMS"])

    # 保存的模型名称
    model_save_name = opt.model_name + "_output" + str(output_size) + "_input" + str(time_step) + "x" + str(input_size) + ".model"

    data_train = list(train_x.numpy().reshape(1,-1, time_step, input_size))
    data_valid = list(valid_x.numpy().reshape(1,-1, time_step, input_size))
    data_train.append(list(train_y.numpy().reshape(-1, 1)))
    data_valid.append(list(valid_y.numpy().reshape(-1, 1)))

    # 最外层是list，次外层是tuple，内层都是ndarray
    data_train = list(zip(*data_train))
    data_valid = list(zip(*data_valid))

    train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=cpu_nums, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=batch_size, num_workers=cpu_nums, pin_memory=True, shuffle=True)

    # 创建网络
    rnn = blstm(input_size, hidden_size, output_size, layers, drop_rate).to(device)
    # 创建优化器
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    # 创建损失函数
    loss_func = nn.CrossEntropyLoss()
    # 定义学习率衰减点，训练到50%和75%时学习率缩小为原来的1/10
    mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch//2, epoch//4*3], gamma=0.1)

    # 训练+验证
    train_loss = []
    valid_loss = []
    min_valid_loss = np.inf
    for i in range(epoch):
        total_train_loss = []    
        rnn.train()     # 进入训练模式
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.type(torch.FloatTensor).to(device)
            b_y = b_y.type(torch.long).to(device)   # CrossEntropy的target是longtensor，且要是1-D，不是one hot编码形式
            prediction = rnn(b_x)                   # rnn output
            # h_s = h_s.data                        # repack the hidden state, break the connection from last iteration
            # h_c = h_c.data                        # repack the hidden state, break the connection from last iteration
            loss = loss_func(prediction[:, -1, :], b_y.view(b_y.size()[0]))         # 计算损失，target要转1-D，注意b_y不是one hot编码形式
            optimizer.zero_grad()                   # clear gradients for this training step
            loss.backward()                         # backpropagation, compute gradients
            optimizer.step()                        # apply gradients
            total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss)) # 存入平均交叉熵
        
        total_valid_loss = [] 
        rnn.eval()      # 进入验证模式
        for step, (b_x, b_y) in enumerate(valid_loader):
            b_x = b_x.type(torch.FloatTensor).to(device) 
            b_y = b_y.type(torch.long).to(device) 
            with torch.no_grad():
                prediction = rnn(b_x)               # rnn output
            # h_s = h_s.data                        # repack the hidden state, break the connection from last iteration
            # h_c = h_c.data                        # repack the hidden state, break the connection from last iteration
            loss = loss_func(prediction[:, -1, :], b_y.view(b_y.size()[0])) 
            total_valid_loss.append(loss.item())
        valid_loss.append(np.mean(total_valid_loss))
        
        if (valid_loss[-1] < min_valid_loss):      
            torch.save({'epoch': i, 'model': rnn, 'train_loss': train_loss, 'valid_loss': valid_loss[-1]}, 
                    os.path.join(model_save_dir, model_save_name)) # 保存字典对象，里面'model'的value是模型 
            min_valid_loss = valid_loss[-1]
            
        # 编写日志
        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((i + 1), epoch,
                                        train_loss[-1],
                                        valid_loss[-1],
                                        min_valid_loss,
                                        optimizer.param_groups[0]['lr'])
        # 学习率更新
        mult_step_scheduler.step()
        # 保存日志
        logger.logger.info(log_string)


