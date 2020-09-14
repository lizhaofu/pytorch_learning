#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学机械学院数控中心
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/9/14 16:28
# @File    : autograd_test.py
# @Software: PyCharm
"""

import torch
# x = torch.ones(2, 2, requires_grad=True)
# print(x)
#
# y = x + 2
# print(y)
# print(y.grad_fn)
#
# z = y * y * 3
# out = z.mean()
# # print(z)
# # print(z, out)
# # print(out)
# #
# # print(out.backward())
# # print(x.grad)
# a = torch.randn(2, 2)
# a = ((a * 3)/(a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)
#
# out.backward()
# # print(x.grad)
x = torch.randn(3, requires_grad=True)
print(x)
y = x * 2
print(y.data.norm())
while y.data.norm() < 1000:
    # print(y)
    y = y * 2
    # print(y.data.norm())

print(y)
v = torch.tensor([0.1, 1.0, 0.0001], dtype= torch.float)
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)


print(x.requires_grad)
y = x.detach()
print(y.requires_grad)

print(x.eq(y).all())