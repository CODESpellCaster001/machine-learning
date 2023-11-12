import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),])
train_set = datasets.MNIST('train_set', # 下载到该文件夹下
                          download=not os.path.exists('train_set'), # 是否下载，如果下载过，则不重复下载
                          train=True, # 是否为训练集
                          transform=transform # 要对图片做的transform
                         )
train_set
test_set = datasets.MNIST('test_set',
                        download=not os.path.exists('test_set'),
                        train=False,
                        transform=transform
                       )
test_set
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

dataiter = iter(train_loader)
images, labels = dataiter.__next__()

print(images.shape)
print(labels.shape)
torch.Size([64, 1, 28, 28])
torch.Size([64])
plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');
#前期准备工作


class NerualNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        """
        定义第一个线性层，
        输入为图片（28x28），
        输出为第一个隐层的输入，大小为128。
        """
        self.linear1 = nn.Linear(28 * 28, 128)
        # 在第一个隐层使用ReLU激活函数
        self.relu1 = nn.ReLU()
        """
        定义第二个线性层，
        输入是第一个隐层的输出，
        输出为第二个隐层的输入，大小为64。
        """
        self.linear2 = nn.Linear(128, 64)
        # 在第二个隐层使用ReLU激活函数
        self.relu2 = nn.ReLU()
        """
        定义第三个线性层，
        输入是第二个隐层的输出，
        输出为输出层，大小为10
        """
        self.linear3 = nn.Linear(64, 10)
        # 最终的输出经过softmax进行归一化
        self.softmax = nn.LogSoftmax(dim=1)

        # 上述动作可以直接使用nn.Sequential写成如下形式：
        self.model = nn.Sequential(nn.Linear(28 * 28, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 10),
                                   nn.LogSoftmax(dim=1)
                                   )

    def forward(self, x):
        """
        定义神经网络的前向传播
        x: 图片数据, shape为(64, 1, 28, 28)
        """
        # 首先将x的shape转为(64, 784)
        x = x.view(x.shape[0], -1)

        # 接下来进行前向传播
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.softmax(x)

        # 上述一串，可以直接使用 x = self.model(x) 代替。

        return x
model = NerualNetwork()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()  # 记录下当前时间
epochs = 15  # 一共训练15轮
for e in range(epochs):
    running_loss = 0  # 本轮的损失值
    for images, labels in train_loader:
        # 前向传播获取预测值
        output = model(images)

        # 计算损失
        loss = criterion(output, labels)

        # 进行反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        # 清空梯度
        optimizer.zero_grad()

        # 累加损失
        running_loss += loss.item()
    else:
        # 一轮循环结束后打印本轮的损失函数
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(train_loader)))

# 打印总的训练时间
print("\nTraining Time (in minutes) =", (time() - time0) / 60)
correct_count, all_count = 0, 0
model.eval() # 将模型设置为评估模式

# 从test_loader中一批一批加载图片
for images,labels in test_loader:
    # 循环检测这一批图片
    for i in range(len(labels)):
        logps = model(images[i])  # 进行前向传播，获取预测值
        probab = list(logps.detach().numpy()[0]) # 将预测结果转为概率列表。[0]是取第一张照片的10个数字的概率列表（因为一次只预测一张照片）
        pred_label = probab.index(max(probab)) # 取最大的index作为预测结果
        true_label = labels.numpy()[i]
        if(true_label == pred_label): # 判断是否预测正确
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
