import os
import sys
import argparse
from flask import Flask, request, jsonify, json

import torch
import torch.nn as nn
import numpy as np

# import 子模块
from utils.logger import *
from utils.parse_config import *
from utils.utils import *

# 创建Flask app
app = Flask(__name__)


'''
post 数据格式:
{
    "keyframes_num": "36",
    "frame_len": "16",
    "skeleton_data": [197.0,228.0,179.0,283.0,...]
}
'''
@app.route("/predict", methods=["POST"])
def predict():
    result_dict = {"sucess": False}
    if request.method == "POST":
        # 获取POST数据
        rec_data = json.loads(request.get_data())
        rec_time_step = int(rec_data["keyframes_num"])
        rec_input_size = int(rec_data["frame_len"])
        if rec_time_step != time_step or \
            rec_input_size != input_size:
            result_dict["error_msg"] = "Input data size error!"
        else:
            rec_skelenton_data = rec_data["skeleton_data"]
            skeleton_data_array = np.array(rec_skelenton_data, dtype=np.float32).reshape(rec_time_step, rec_input_size)
            # 按每帧转换为相对坐标
            # for i in range(rec_time_step):
            #     skeleton_data_array[i] = abs2rel(skeleton_data_array[i], crop_size)
            # 转换为tensor
            skeleton_data_tensor = torch.from_numpy(skeleton_data_array).to(device).unsqueeze(0)
            # 计算预测结果
            with torch.no_grad():
                prediction = model(skeleton_data_tensor)
                pre_result = torch.max(prediction[:, -1, :], 1)[1].cpu().data.numpy().tolist()[0]
            pre_class_name = class_index2name(class_dict, pre_result, dict_start_index)
            result_dict["prediction"] = pre_class_name
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
    model_save_name = opt.model_name + "_output" + str(output_size) + "_input" + str(time_step) + "×" + str(input_size) + ".model"
    model_save_path = os.path.join(model_save_dir, model_save_name)
    # 判断模型文件是否存在
    if not os.path.exists(model_save_path):
        logger.logger.error("model file is not existed!")
        exit()
    
    # 加载模型
    model = torch.load(model_save_path).get('model')
    if use_gpu:
        model = model.cuda()
    model.eval()

    app.run(host='0.0.0.0', port=5000, debug=True)