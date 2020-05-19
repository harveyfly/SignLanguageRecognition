import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

from nnet.AutoEncoder import *


# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
N_TEST_IMG = 5
TIME_STEP = 20
INPUT_SIZE = 28
OUTPUT_SIZE = 3

dataset_dir = 'data\SLR_dataset\processed'
# DataLoader
np_data_x = np.load(os.path.join(dataset_dir, 'SLR_S0_E20_body_data.npy'), allow_pickle=True)
np_data_y = np.load(os.path.join(dataset_dir, 'SLR_S0_E20_body_label.npy'), allow_pickle=True)
data_x = torch.from_numpy(np_data_x)
data_y = torch.from_numpy(np_data_y)
print(data_x.shape, data_y.shape)

data_train = list(data_x.numpy().reshape(1, -1, TIME_STEP, INPUT_SIZE))
data_train.append(list(data_y.numpy().reshape(-1, 1)))
data_train = list(zip(*data_train))
train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=True)

# Autoencoder
autoencoder = AutoEncoder(INPUT_SIZE, TIME_STEP, OUTPUT_SIZE)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

# Train
for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):
        b_x = x.view(-1, TIME_STEP*INPUT_SIZE)
        b_y = x.view(-1, TIME_STEP*INPUT_SIZE)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

            # plotting decoded image (second row)
            # _, decoded_data = autoencoder(view_data)
            # for i in range(N_TEST_IMG):
            #     a[1][i].clear()
            #     a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
            #     a[1][i].set_xticks(()); a[1][i].set_yticks(())
            # plt.draw(); plt.pause(0.05)

# plt.ioff()
# plt.show()

# visualize in 3D plot
view_data = data_x[:100]
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = data_y[:100].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/20)); ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()
