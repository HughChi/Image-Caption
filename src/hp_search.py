# -*- coding: utf-8 -*-
from __future__ import print_function

import os
from math import log

import keras
from hyperas import optim#Hyperas可以在Keras中自动选择超参数
from hyperas.distributions import loguniform
from hyperas.distributions import uniform
from hyperopt import Trials, STATUS_OK, tpe
from keras.layers import Input, LSTM, Concatenate, Embedding, RepeatVector, TimeDistributed
from keras.layers.core import Dense, Dropout
from keras.models import Model

from config import batch_size, num_train_samples, num_valid_samples, max_token_length, vocab_size, embedding_size, \
    best_model
from data_generator import DataGenSequence


def data():
    return DataGenSequence('train'), DataGenSequence('valid')#返回训练集序列和验证集序列


def create_model():
    # word embedding
    text_input = Input(shape=(max_token_length,), dtype='int32')#输入行数为max_token_length的矩阵，数据类型是32位整数
    x = Embedding(input_dim=vocab_size, output_dim=embedding_size)(text_input)#embedding层
    x = LSTM(256, return_sequences=True)(x)#输入LSTM模型,256个神经元
    text_embedding = TimeDistributed(Dense(embedding_size))(x)#使用包装器TimeDistributed包装Dense，以产生针对各个时间步信号的独立全连接

    # image embedding
    image_input = Input(shape=(2048,))#输入行数为2048的图形数据矩阵
    #全连接层dense，激活函数：relu
    x = Dense(embedding_size, activation='relu', name='image_embedding')(image_input)
    # the image I is only input once RepeatVector层将输入只重复1次
    image_embedding = RepeatVector(1)(x)

    # language model
    x = [image_embedding, text_embedding]
    #按列合并
    x = Concatenate(axis=1)(x)
    #以一定概率丢弃神经元，uniform随机生成0-1内一个实数
    x = Dropout({{uniform(0, 1)}})(x)
    #输入LSTM模型,1024个神经元
    x = LSTM(1024, return_sequences=True, name='language_lstm_1')(x)
    x = Dropout({{uniform(0, 1)}})(x)
    x = LSTM(1024, name='language_lstm_2')(x)
    x = Dropout({{uniform(0, 1)}})(x)
    # 全连接层dense，激活函数：softmax
    output = Dense(vocab_size, activation='softmax', name='output')(x)

    inputs = [image_input, text_input]
    model = Model(inputs=inputs, outputs=output)
    model_weights_path = os.path.join('models', best_model)
    #加载权值
    model.load_weights(model_weights_path)
    #Adam 优化器
    adam = keras.optimizers.Adam(lr={{loguniform(log(1e-6), log(1e-3))}})
    #将优化器传递给 model.compile()，损失函数为多分类的对数损失函数
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    #逐个生成数据的batch并进行训练
    model.fit_generator(
        DataGenSequence('train'),
        steps_per_epoch=num_train_samples / batch_size // 10,
        validation_data=DataGenSequence('valid'),
        validation_steps=num_valid_samples / batch_size // 10)

    score, acc = model.evaluate_generator(DataGenSequence('valid'), verbose=0)#使用一个生成器作为数据源，来评估模型
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,#algo参数指定搜索算法，tpe表示 tree of Parzen estimators
                                          max_evals=10,#执行的最大评估次数max_evals
                                          trials=Trials())#在每个时间步存储信息

    print("Evalutation of best performing model:")#最佳表现模型评估
    print(best_model.evaluate_generator(DataGenSequence('valid')))#使用一个生成器作为数据源，来评估模型
    print("Best performing model chosen hyper-parameters:")#表现最佳的模型选择的超参数：
    print(best_run)
