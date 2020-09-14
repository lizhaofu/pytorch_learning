#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学机械学院数控中心
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/9/14 18:37
# @File    : nn_learning.py
# @Software: PyCharm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_feature = 1
        for s in size:
            num_feature *= s
        return num_feature



net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())


# input = torch.randn(1, 1, 32, 32)
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))


output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

# print(loss.creator) # MSELoss
# print(loss.creator.previous_functions[0][0]) # Linear
# print(loss.creator.previous_functions[0][0].previous_functions[0][0]) # ReLU

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)
#
# print(list(net.parameters())[0])

optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
print(loss)
optimizer.step()