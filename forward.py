# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')
#引用keras抽象后端引擎
import keras.backend as K
#引入tensorflow
import tensorflow as tf
#从keras中引入输入层、全连接层、LSTM层、Concatenate层、嵌入层、输入重复机制、TimeDistributed层、Dropout层
from keras.layers import Input, Dense, LSTM, Concatenate, Embedding, RepeatVector, TimeDistributed, Dropout
#引入模型层
from keras.models import Model
#引入模型结构绘制机制
from keras.utils import plot_model

from config import max_token_length
from config import vocab_size, embedding_size


def build_model():
    # 输入文本信息
    text_input = Input(shape=(max_token_length,), dtype='int32')
	# 将文本信息转化为词向量
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size)(text_input)
	# 将转换后的词向量输入LSTM网络
    x = LSTM(256, return_sequences=True)(x)
	# 使用包装器TimeDistributed包装Dense，以产生针对各个时间步信号的独立全连接
    text_embedding = TimeDistributed(Dense(embedding_size))(x)

    # 输入image embedding
    image_input = Input(shape=(2048,))
    x = Dense(embedding_size, activation='relu', name='image_embedding')(image_input)
    # the image I is only input once 每一张图片被输入一次
    image_embedding = RepeatVector(1)(x)

    # language model
	# 将image embedding和 text embedding合并到一起
    x = [image_embedding, text_embedding]
	#将image embedding与text embedding按第一维度拼接起来
    x = Concatenate(axis=1)(x)
	#以0.1的概率丢弃信息
    x = Dropout(0.1)(x)
	#输入LSTM层
    x = LSTM(1024, return_sequences=True, name='language_lstm_1')(x)
	#以0.2的概率丢弃信息
    x = Dropout(0.2)(x)
	#输入LSTM层
    x = LSTM(1024, name='language_lstm_2')(x)
	#以0.4的概率丢弃信息
    x = Dropout(0.4)(x)
	#经过全连接层后输出
    output = Dense(vocab_size, activation='softmax', name='output')(x)
    #定义输入
    inputs = [image_input, text_input]
	#传入模型
    model = Model(inputs=inputs, outputs=output)
    return model


if __name__ == '__main__':
    #使用cpu运行模型
    with tf.device("/cpu:0"):
        model = build_model()
	#打印模型信息
    print(model.summary())
	#打印模型结构
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()