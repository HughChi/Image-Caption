# -*- coding: utf-8 -*-
import multiprocessing

import cv2 as cv
import tensorflow as tf
from tensorflow.python.client import device_lib


# 获取GPU的数量
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# 获取CPU的数量
def get_available_cpus():
    return multiprocessing.cpu_count()

# 在图片上写文字用的方法  将字符写到图片上输出
def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

# 使用交叉熵损失函数作为分类损失函数
def sparse_loss(y_true, y_pred):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)