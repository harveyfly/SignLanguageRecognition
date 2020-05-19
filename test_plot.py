from utils.plot_data import *
import os

data_dir = "data\SLR_dataset\processed"
data_path = os.path.join(data_dir, "SLR_S0_E20_body_data.npy")
label_path = os.path.join(data_dir, "SLR_S0_E20_body_label.npy")
# 读取数据
np_data = np.load(data_path, allow_pickle=True)
np_label = np.load(label_path, allow_pickle=True)

# 选取标签
select_idx = 0
for i in range(len(np_data)):
    if np_label[i] == select_idx:
        print(np_label[i])
        plot_data(np_data[i], need_all=False)

# mat_data_dir = "./data/SLR_dataset/xf500_body_depth_mat/000000"
# for file in os.listdir(mat_data_dir):
#     mat_data_path = os.path.join(mat_data_dir, file)
#     # 读取mat数据
#     mat_data = scio.loadmat(mat_data_path)['data'].astype(np.float32)
#     plot_data(mat_data, need_all=True)