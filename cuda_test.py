#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学机械学院数控中心
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/9/14 16:07
# @File    : cuda_test.py
# @Software: PyCharm
"""
import torch
# x = torch.randn(4, 4)
x = torch.randn(1)
print(x)
print(x.item())
if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)
    # x = x.to(device)
    # z = x + y
    # print(z)
    # print(z.to('cpu', torch.double))
    x = x.cuda()
    y = y.cuda()
    z = x + y
    print(z)