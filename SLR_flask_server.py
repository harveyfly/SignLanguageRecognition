import os
import sys
import argparse
from flask import Flask, request, jsonify, json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import 子模块
from utils.logger import *
from utils.parse_config import *
from utils.utils import *
from utils.keyframes import *

# 创建Flask app
app = Flask(__name__)


'''
post 数据格式:
{
    "keyframes_num": "36",
    "frame_len": "24",
    "skeleton_data": [197.0,228.0,179.0,283.0,...]
}
'''
@app.route("/predict", methods=["POST"])
def predict():
    result_dict = {"sucess": False}
    if request.method == "POST":
        # 获取POST数据
        rec_data = json.loads(request.get_data())
        rec_input_size = int(rec_data["frame_len"])
        if rec_input_size != input_size:
            result_dict["error_msg"] = "Input data size error!"
        else:
            rec_skelenton_data = rec_data["skeleton_data"]
            skeleton_data_array = np.array(rec_skelenton_data, dtype=np.float32).reshape(-1, rec_input_size)
            # 提取关键帧
            key_indexes = extract_keyframes_indexes(skeleton_data_array, time_step)
            if len(key_indexes) < time_step:
                return jsonify(result_dict)
            skeleton_data_array = skeleton_data_array[key_indexes]
            # 按每帧转换为相对坐标
            for i in range(rec_time_step):
                skeleton_data_array[i] = abs2rel(skeleton_data_array[i], crop_size)
            # 转换为tensor
            skeleton_data_tensor = torch.from_numpy(skeleton_data_array).to(device).unsqueeze(0)
            # 计算预测结果
            with torch.no_grad():
                prediction = model(skeleton_data_tensor)
                pre_result = torch.max(F.softmax(prediction[:, -1, :], dim=1), 1)
                pre_class = pre_result[1].cpu().data.numpy().tolist()
                pre_prob = pre_result[0].cpu().data.numpy().tolist()
            pre_class_name = class_index2name(class_dict, pre_class, dict_start_index)
            if pre_prob > 0.9:
                result_dict["prediction"] = pre_class_name
            else:
                result_dict["prediction"] = "Unknown"
            result_dict["sucess"] = True
        return jsonify(result_dict)
    
@app.route("/getSysParameter", methods=["GET"])
def getSysParameter():
    if request.method == "GET":
        return jsonify({
            "sucess": True,
            "keyframes_num": time_step,
            "frame_len": input_size,
            "crop_size": crop_size
        })
    else:
        return jsonify({
            "sucess": False,
            "error_msg": "Http method error"
        })

# 加载模型
def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="config/Net.cfg", help="path to model definition file")
    parser.add_argument("--model_name", type=str, default="blstm", help="used model name (lstm, blstm)")
    parser.add_argument("--server_config", type=str, default="config/SLR_server.cfg", help="path to server config file")
    parser.add_argument("--crop_size", type=int, default=256, help="size of each crop image")
    opt = parser.parse_args()
    print(opt)

    # 读取配置文件
    server_config = parse_data_config(opt.server_config)
    model_config = parse_model_config(opt.model_config)[opt.model_name]

    # 图片裁剪大小
    crop_size = int(opt.crop_size)

    # 记录日志
    logger = Logger(server_config["log_path"])

    # 读取字典
    dict_path = server_config["dictionary_path"]
    if not os.path.exists(dict_path):
        logger.logger.error("class dict is not exist")
        exit()
    class_dict = read_dict_table(dict_path)
    dict_start_index = int(server_config["dict_start_index"])

    # 设置GPU
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    logger.logger.info(str(device))
    # 获取gpu名字
    logger.logger.info(torch.cuda.get_device_name(0))

    # 设置随机种子
    torch.manual_seed(int(model_config["SEED"]))

    # 处理模型参数
    batch_size = int(model_config["BATCH_SIZE"])
    cpu_nums = int(model_config["CPU_NUMS"])
    time_step = int(model_config["TIME_STEP"])
    input_size = int(model_config["INPUT_SIZE"])
    output_size = int(model_config["OUTPUT_SIZE"])

    # 保存的模型名称
    model_save_dir = server_config["model_save_dir"]
    model_save_name = opt.model_name + "_output" + str(output_size) + "_input" + str(time_step) + "x" + str(input_size) + ".pkl"
    model_save_path = os.path.join(model_save_dir, model_save_name)
    # 判断模型文件是否存在
    if not os.path.exists(model_save_path):
        logger.logger.error("model file is not existed!")
        exit()
    
    # 加载模型
    model = load_checkpoint(model_save_path, device)
    if use_gpu:
        model.to(device)

    app.run(host='0.0.0.0', port=61504, debug=True)
