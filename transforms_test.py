#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学机械学院数控中心
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/9/14 08:59
# @File    : transforms_test.py
# @Software: PyCharm
"""
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


img_path = "hymenoptera_data/train/ants/5650366_e22b7e1065.jpg"



img = Image.open(img_path)

writer = SummaryWriter('logs')


tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)
print(tensor_img)
