# -*- coding: utf-8 -*-
import pickle
from tqdm import tqdm
import keras
import numpy as np
from keras.preprocessing import sequence
from keras.utils import Sequence

from config import batch_size, max_token_length, vocab_size


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        vocab = pickle.load(open('data/vocab_train.p', 'rb'))
        #反序列化词汇表，将文件中的数据解析为一个python对象
        self.idx2word = sorted(vocab)#对vocab排序
        self.word2idx = dict(zip(self.idx2word, range(len(vocab))))
        #将排序好的列表和从0开始的序号列表打包成元组，并对应生成字典

        filename = 'data/encoded_{}_images.p'.format(usage)#用usage替换占位符{}中的内容
        self.image_encoding = pickle.load(open(filename, 'rb'))#反序列化图像编码

        if usage == 'train':
            samples_path = 'data/samples_train.p'#训练集路径
        else:
            samples_path = 'data/samples_valid.p'#验证集路径

        samples = pickle.load(open(samples_path, 'rb'))
        self.samples = samples
        np.random.shuffle(self.samples)#打乱

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))
        #返回大于等于samples长度/bactch size的最小整数

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.samples) - i))
        #batch size和sample长度-i中的小值作为length
        batch_image_input = np.empty((length, 2048), dtype=np.float32)
        #生成一个随机元素的矩阵，行数为length，列数为2048，数据类型为32位浮点数
        batch_y = np.empty((length, vocab_size), dtype=np.int32)
        #生成一个随机元素的矩阵，行数为length，列数为vocab_size，数据类型为32位浮点数
        text_input = []

        for i_batch in tqdm(range(length)):
            sample = self.samples[i + i_batch]
            image_id = sample['image_id']#读取image_id键对应值
            image_input = np.array(self.image_encoding[image_id])#生成对应图像二维数组
            text_input.append(sample['input'])#向text_input列表添加sample中input键对应值
            batch_image_input[i_batch] = image_input
            batch_y[i_batch] = keras.utils.to_categorical(sample['output'], vocab_size)
            #将output整型标签转为onehot，vocab_size为标签类别总数

        batch_text_input = sequence.pad_sequences(text_input, maxlen=max_token_length, padding='post')#将text_input序列转化为经过填充的等长的新序列，max_token_length为序列的最大长度，当需要补0时在序列的结尾补
        return [batch_image_input, batch_text_input], batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.samples)#打乱顺序


def train_gen():
    return DataGenSequence('train')#生成训练集序列


def valid_gen():
    return DataGenSequence('valid')#生成验证集序列
