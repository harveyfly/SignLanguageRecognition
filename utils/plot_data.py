import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

# 所有骨骼节点
skeletons_all = [
    (0, 1),
    (0, 12),
    (0, 16),
    (1, 20),
    (3, 20),
    (2, 3),
    (4, 20),
    (4, 5),
    (5, 6),
    (6, 7),
    (6, 22),
    (7, 21),
    (8, 20),
    (8, 9),
    (9, 10),
    (10, 11),
    (10, 24),
    (11, 23),
    (12, 13),
    (13, 14),
    (14, 15),
    (16, 17),
    (17, 18),
    (18, 19)
]

# # 选取的骨骼节点
# skeletons_need = [
#     (0, 9),
#     (1, 9),
#     (5, 9),
#     (1, 2),
#     (5, 6),
#     (2, 3),
#     (6, 7),
#     (3, 4),
#     (7, 8),
#     (3, 11),
#     (7, 13),
#     (4, 10),
#     (8, 12)
# ]

# 选取的骨骼节点
skeletons_need = [
    (0, 1),
    (4, 5),
    (1, 2),
    (5, 6),
    (2, 3),
    (6, 7)
]


# 绘图测试
def plot_data(data, need_all=False):
    # 打开交互模式
    plt.ion()
    plt.show()
    for i in range(len(data)):
        joint_frame = data[i]
        x_data = joint_frame[0::2]
        y_data = joint_frame[1::2]
        # 清除原有图像
        plt.cla()
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        plt.scatter(x_data, y_data)
        if need_all == True:
            skeletons = skeletons_all
        else:
            skeletons = skeletons_need
        for skeleton in skeletons:
            s1 = skeleton[0]
            s2 = skeleton[1]
            a = [x_data[s1], x_data[s2]]
            b = [y_data[s1], y_data[s2]]
            plt.plot(a, b, color='r')

        plt.pause(0.1)