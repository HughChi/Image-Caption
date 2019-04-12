# -*- coding: utf-8 -*-
import os

img_rows, img_cols, img_size = 224, 224, 2243#图像行数，列数，大小
channel = 3#通道数
batch_size = 256#batch大小
epochs = 10000#迭代次数
patience = 50#容忍度
num_train_samples = 14883151#训练样本数
num_valid_samples = 2102270#验证样本数
embedding_size = 128#嵌入层数目
vocab_size = 17628#词汇数目
max_token_length = 40#分词后token的最大长度
num_image_features = 2048#图像特征数
hidden_size = 512# 隐藏层中单元数目

train_folder = 'data/ai_challenger_caption_train_20170902'#训练集路径
valid_folder = 'data/ai_challenger_caption_validation_20170910'#验证集路径
test_a_folder = 'data/ai_challenger_caption_test_a_20180103'#测试集a路径
test_b_folder = 'data/ai_challenger_caption_test_b_20180103'#测试集b路径
train_image_folder = os.path.join(train_folder, 'caption_train_images_20170902')#训练集图像路径
valid_image_folder = os.path.join(valid_folder, 'caption_validation_images_20170910')#验证集图像路径
test_a_image_folder = os.path.join(test_a_folder, 'caption_test_a_images_20180103')#测试集a图像路径
test_b_image_folder = os.path.join(test_b_folder, 'caption_test_b_images_20180103')#测试集b图像路径
train_annotations_filename = 'caption_train_annotations_20170902.json'#训练集标注路径
valid_annotations_filename = 'caption_validation_annotations_20170910.json'#验证集标注路径
test_a_annotations_filename = 'caption_test_a_annotations_20180103.json'#测试集a标注路径
test_b_annotations_filename = 'caption_test_b_annotations_20180103.json'#测试集b标注路径

start_word = '<start>'#开始的词
stop_word = '<end>'#结束的词
unknown_word = '<UNK>'#未知词

best_model = 'model.04-1.3820.hdf5'
beam_size = 20#光束大小
