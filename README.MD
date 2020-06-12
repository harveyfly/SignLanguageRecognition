# 项目介绍
+ 该项目的目标是：**基于PyTorch实现中文孤立手语词识别**；
+ 目前使用的基本网络结构是：**Bi-LSTM**（双向长短期记忆神经网络）；
+ 数据集使用的是：中国科学技术大学提供的**500-CSL**（500类中文手语单词数据集）[http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html)；
# 项目文件树
整个项目工程结构如下（Git默认设置不能跟踪空文件夹，需要手动创建）
```
SignLanguageRecognition
 ├── config
 │   ├── Net.cfg
 │   ├── SLR_dataset.cfg
 │   ├── SLR_server.cfg
 │   └── __init__.py
 ├── data
 │   └── SLR_dataset
 │       ├── dictionary.txt
 │       ├── processed/
 │       ├── txt2mat.m
 │       ├── xf500_body_depth_mat/
 │       ├── xf500_body_depth_mat.zip
 │       ├── xf500_body_depth_txt/
 │       ├── xf500_body_depth_txt.zip
 │       ├── xf500_color_video/
 │       └── xf500_depth_video/
 ├── data_preprocess.py
 ├── log
 │   ├── SLR_output.log
 │   └── SLR_server.log
 ├── model
 │   └── SLR
 │       └── blstm_output50_input36x24.pkl
 ├── nnet
 │   ├── AutoEncoder.py
 │   ├── blstm.py
 │   ├── lstm.py
 │   └── __init__.py
 ├── requirements.txt
 ├── SLR_flask_server.py
 ├── test.py
 ├── test_server.http
 ├── train.py
 └── utils
     ├── keyframes.py
     ├── logger.py
     ├── parse_config.py
     ├── plot_data.py
     ├── utils.py
     └── __init__.py
```
# 环境配置
+ Ubuntu 18.04 / Windows 10
+ VSCode + Python 3.7
+ PyTorch 1.2.0
# 操作流程
## 安装软件包
```shell
$ pip install -r requirements.txt
```
## 下载数据集
### 方式一：
+ 使用我的数据集备份（已经转化为mat文件），链接：[https://pan.baidu.com/s/1lZasotYbZpsoCrilvT71VQ](https://pan.baidu.com/s/1lZasotYbZpsoCrilvT71VQ)，提取码：vswo
### 方式二：
+ 在官方数据集下载`xf500_body_depth_txt.zip`，[https://pan.baidu.com/s/1tdUdI_3Ius44__d9PzVRCA#list/path=%2F](https://pan.baidu.com/s/1tdUdI_3Ius44__d9PzVRCA#list/path=%2F)
+ 使用MATLAB执行`txt2mat.m`将txt文件转化为mat文件 
## 数据预处理
```shell
$ python data_preprocess.py
```
可以在源代码中修改参数，生成数据在`data\SLR_dataset\processed`
## 修改配置文件
### 网络配置
+ Net.cfg
```
[blstm]
# 定义输入特征数
INPUT_SIZE = 24
# 定义一个LSTM单元有多少个神经元
HIDDEN_SIZE = 1024
# 定义输出种类数
OUTPUT_SIZE = 50
# 批大小
BATCH_SIZE = 128
# 学习次数
EPOCH = 20
# 学习率
LEARNING_RATE = 0.001
# 时间步长
TIME_STEP = 36
# drop out 概率
DROP_RATE = 0.5
# 隐藏层数量
LAYERS = 2
# 随机种子
SEED = 0
# CPU线程数
CPU_NUMS = 32
```
### 数据文件配置
+ SLR_dataset.cfg
```
# 文件路径
dataset_dir = ./data/SLR_dataset/processed
# 数据文件名
data_file_name = SLR_S45_E95_K36_body_data.npy
# 标签文件名
label_file_name = SLR_S45_E95_K36_body_label.npy
# 训练集大小
train_data_size = 0.8
# 验证集大小
valid_data_size = 0.1
# 测试集大小
test_data_size = 0.1
# 模型保存文件夹
model_save_dir = ./model/SLR
# 日志保存文件
log_path = ./log/SLR_output.log
```
## 训练
```shell
$ python train.py
```
## 测试
```shell
$ python test.py
```
训练和测试结果记录在日志文件（`log/`）中，终端也会显示；模型文件保存在`model/`文件夹下
## 服务端运行
```shell
$ python SLR_flask_server.py
```
服务端接口设计较为简单，仅需要1个GET操作和1个POST操作即可，GET操作用来请求系统参数列表，POST用来连接手语词识别的方法，得到数据预测的识别结果。
|接口名称|调用方法|请求参数|返回值说明|
|:----:|:----:|:----:|:----:|
|getSysParameter|GET|-|success：表示操作是否成功；keyframes_num：关键帧数量；frame_len：帧长度；crop_size：图片裁剪大小；|
|predict|POST|keyframes_num：关键帧数量；frame_len：帧长度；skeleton_data：骨骼关节点位置数据，大小约束为keyframes_num×frame_len；|success：表示操作是否成功；prediction：预测结果；|

