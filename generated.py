# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')
import json
import pickle
import zipfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import jieba
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import (load_img, img_to_array)
from tqdm import tqdm

from config import img_rows, img_cols
from config import start_word, stop_word, unknown_word
from config import train_annotations_filename
from config import train_folder, valid_folder, test_a_folder, test_b_folder
from config import train_image_folder, valid_image_folder, test_a_image_folder, test_b_image_folder
from config import valid_annotations_filename

#调用Keras中的ResNet50模型，加载在ImageNet ILSVRC比赛中已经训练好的权重
#include_top表示是否包含模型顶部的全连接层，如果不包含，则可以利用这些参数来做一些定制的事情
image_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')

#确定是否存在文件夹
def ensure_folder(folder):
    #如果不存在文件夹，创建文件夹
    if not os.path.exists(folder):
        os.makedirs(folder)

#解压文件
def extract(folder):
    #folder.zip
    filename = '{}.zip'.format(folder)
    #输出解压名称并执行解压操作
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')

#将图像文件编码
def encode_images(usage):
    encoding = {}
    #编码训练集
    if usage == 'train':
        image_folder = train_image_folder
    #编码验证集
    elif usage == 'valid':
        image_folder = valid_image_folder
    #编码测试集a
    elif usage == 'test_a':
        image_folder = test_a_image_folder
    #编码测试集b
    else:  # usage == 'test_b':
        image_folder = test_b_image_folder
    #batch_size为256
    batch_size = 256
    #names储存文件夹中所有的jpg文件名称
    names = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    #计算一共多少批次，ceil为向上取整
    num_batches = int(np.ceil(len(names) / float(batch_size)))

    #输出编码过程
    print('ResNet50提取特征中...')
    #对每个batche进行处理，使用tqdm库显示处理进度
    for idx in range(num_batches):
        #该批次开始的位置
        i = idx * batch_size
        #该批次的长度，会出现最后一个批次不够batchsize的情况
        length = min(batch_size, (len(names) - i))
        #使用empty创建一个多维数组
        image_input = np.empty((length, img_rows, img_cols, 3))
        #对于每一张图片
        for i_batch in range(length):
            #提取图片名称
            image_name = names[i + i_batch]
            #提取路径名称
            filename = os.path.join(image_folder, image_name)
            #keras读取图片，并且将图片调整为224*224
            img = load_img(filename, target_size=(img_rows, img_cols))
            #将图片转为矩阵
            img_array = img_to_array(img)
            #使用keras内置的preprocess_input进行图片预处理，默认使用caffe模式去均值中心化
            img_array = keras.applications.resnet50.preprocess_input(img_array)
            #将处理后的图片保存到image_input中
            image_input[i_batch] = img_array

        #使用ResNet50网络进行预测，预测结果保存到preds中
        preds = image_model.predict(image_input)

        #对于每一张图片
        for i_batch in range(length):
            #提取图片名称
            image_name = names[i + i_batch]
            #把预测结果保存到encoding中
            encoding[image_name] = preds[i_batch]

    #用相应的类别命名
    filename = 'data/encoded_{}_images.p'.format(usage)
    #使用python的pickle模块把数据进行序列化，把encoing保存到filename中
    with open(filename, 'wb') as encoded_pickle:
        pickle.dump(encoding, encoded_pickle)
    print('ResNet50提取特征完毕...')

#处理数据集的标注部分，生成训练集的词库
def build_train_vocab():
    #提取训练集标注文件的路径
    #data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json
    annotations_path = os.path.join(train_folder, train_annotations_filename)

    #读取json格式的标注文件
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    #输出处理进度
    print('building {} train vocab')
    #创建一个无序不重复元素集
    vocab = set()
    #使用tqdm输出进度
    for a in tqdm(annotations):
        #提取annotations每一行的caption注释
        caption = a['caption']
        #对于每一个caption
        for c in caption:
            #使用jieba进行分词
            seg_list = jieba.cut(c)
            #把每个词加入到vocab中
            for word in seg_list:
                vocab.add(word)
    #在vocab中加入<start><stop><UNK>
    vocab.add(start_word)
    vocab.add(stop_word)
    vocab.add(unknown_word)

    #将vocab写入vocab_train.p
    filename = 'data/vocab_train.p'
    with open(filename, 'wb') as encoded_pickle:
        pickle.dump(vocab, encoded_pickle)

#创建samples
def build_samples(usage):
    #如果进行训练
    if usage == 'train':
        #路径为train_folder
        annotations_path = os.path.join(train_folder, train_annotations_filename)
    else:
        #否则路径为valid_folder
        annotations_path = os.path.join(valid_folder, valid_annotations_filename)
    with open(annotations_path, 'r') as f:
        #同时加载json文件
        annotations = json.load(f)

    #将vocab文件反序列化提取词汇
    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
    #index to word 对vocab进行排序
    idx2word = sorted(vocab)
    #word to index zip函数将idx2word与序号索引打包为元祖，用dict函数将映射关系构造为字典，词：索引
    word2idx = dict(zip(idx2word, range(len(vocab))))

    #输出进度信息
    print('building {} samples'.format(usage))
    #列表samples
    samples = []
    #对于每一项annotation
    for a in tqdm(annotations):
        #提取image_id
        image_id = a['image_id']
        #提取caption
        caption = a['caption']
        #对于每一项caption
        for c in caption:
            #使用jieba进行分词
            seg_list = jieba.cut(c)
            #列表inpit
            input = []
            #last_word标签设置为start
            last_word = start_word
            #使用enumerate函数列出下标和数据
            for j, word in enumerate(seg_list):
                #如果词库中没有word
                if word not in vocab:
                    #word修改为UNK
                    word = unknown_word
                #input添加序号
                input.append(word2idx[last_word])
                #samples添加id，input，output
                samples.append({'image_id': image_id, 'input': list(input), 'output': word2idx[word]})
                #last_word设置成word
                last_word = word
            #input添加last_word
            input.append(word2idx[last_word])
            #samples添加id，input，stop_word
            samples.append({'image_id': image_id, 'input': list(input), 'output': word2idx[stop_word]})

    #打包samples信息
    filename = 'data/samples_{}.p'.format(usage)
    with open(filename, 'wb') as f:
        pickle.dump(samples, f)


#主函数
if __name__ == '__main__':
    # parameters
    # 确定是否存在data
    ensure_folder('data')

    #解压文件
    # if not os.path.isdir(train_image_folder):
    #extract(train_folder)

    #解压文件
    # if not os.path.isdir(valid_image_folder):
    #extract(valid_folder)

    #解压文件
    # if not os.path.isdir(test_a_image_folder):
    #extract(test_a_folder)

    #解压文件
    # if not os.path.isdir(test_b_image_folder):
    #extract(test_b_folder)

    #编码train
    if not os.path.isfile('data/encoded_train_images.p'):
        encode_images('train')

    #编码valid
    if not os.path.isfile('data/encoded_valid_images.p'):
        encode_images('valid')

    #编码test_a
    if not os.path.isfile('data/encoded_test_a_images.p'):
        encode_images('test_a')

    #编码test_b
    if not os.path.isfile('data/encoded_test_b_images.p'):
        encode_images('test_b')

    #生成词库
    if not os.path.isfile('data/vocab_train.p'):
        build_train_vocab()

    #生成train的图片与标注数据
    if not os.path.isfile('data/samples_train.p'):
        build_samples('train')

    #生成valid的图片与标注数据
    if not os.path.isfile('data/samples_valid.p'):
        build_samples('valid')
        
def test_gen():
    encode_images('test_a')
