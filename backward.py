# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')
import argparse

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

from config import patience, epochs, num_train_samples, num_valid_samples, batch_size
from data_generator import train_gen, valid_gen
from forward import build_model
from utils import get_available_gpus, get_available_cpus

#主函数
if __name__ == '__main__':
    # Parse arguments
    #创建一个ArgumentParser实例
    ap = argparse.ArgumentParser()
    #添加参数-p
    ap.add_argument("-p", "--pretrained", help="path to save pretrained model files")
    # 把parser中设置的所有"add_argument"给返回到ap子类实例当中
    #vars() 函数返回对象object的属性和属性值的字典对象
    args = vars(ap.parse_args())
    #获取当前路径
    pretrained_path = args["pretrained"]
    #模型checkpoint路径
    checkpoint_models_path = 'models/'

    # Callbacks
    #回调函数
    #使用TensorBoard可视化训练曲线
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    #模型的命名规则
    model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.hdf5'
    #使用ModelCheckpoint保存训练模型，monitor需要监视的值
    #verbose进度条信息展示模式，save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    #使用early_stop防止过拟合，monitor: 被监测的数据。patience：没有进步的训练轮数，在这之后训练就会被停止。
    early_stop = EarlyStopping('val_loss', patience=patience)
    #使用函数调整学习率，当评价指标不在提升时，减少学习率
    #factor：每次减少学习率的因子，lr = lr*factor
    #当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 5), verbose=1)

    #子类MyCbk，创建一个自定义的回调函数
    #用来组建新的回调函数的抽象基类
    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            #定义要保存的模型
            self.model_to_save = model

        #定义保存模型函数，传入epoch和logs，并命名模型
        #在每轮结束时被调用。
        def on_epoch_end(self, epoch, logs=None):
            fmt = checkpoint_models_path + 'model.%02d-%.4f.hdf5'
            self.model_to_save.save(fmt % (epoch, logs['val_loss']))


    # Load our model, added support for Multi-GPUs
    #调用build_model()函数创建模型
    #检测是否可以用GPU训练
    num_gpu = len(get_available_gpus())
    if num_gpu >= 2:
        #使用CPU创建模型
        with tf.device("/cpu:0"):
            model = build_model()
            #如果存在预训练模型
            if pretrained_path is not None:
                #从HDF5文件中加载权重到当前模型中，by_name=True，只有名字匹配的层才会载入权重
                model.load_weights(pretrained_path, by_name=True)
        #使用多GPU并行训练
        new_model = multi_gpu_model(model, gpus=num_gpu)
        # rewrite the callback: saving through the original model and not the multi-gpu model.
        #保存模型
        model_checkpoint = MyCbk(model)
    else:
        #创建模型
        new_model = build_model()
        #如果存在预训练模型
        if pretrained_path is not None:
            #从HDF5文件中加载权重到当前模型中
            new_model.load_weights(pretrained_path)

    #使用Adam优化器，学习率为0.00005
    adam = keras.optimizers.Adam(lr=5e-5)
    #自定义损失函数，使用adam优化器，损失函数为categorical_crossentropy多分类对数损失
    #metrics评价函数，在训练和测试期间的模型评估标准。
    new_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    #summary():打印出模型概况
    print(new_model.summary())

    # Final callbacks
    #回调函数是一个函数的合集，会在训练的阶段中所使用。
    #你可以使用回调函数来查看训练模型的内在状态和统计。
    #ModelCheckpoint在每个训练期之后保存模型。
    #EarlyStopping当被监测的数量不再提升，则停止训练。
    #ReduceLROnPlateau当标准评估已经停止时，降低学习速率。
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    #使用fit_generator进行训练
    #batch_size: 整数或 None。每次梯度更新的样本数。如果未指定，默认为 32
    #epochs: 整数。训练模型迭代轮次。一个轮次是在整个 x 和 y 上的一轮迭代。
    #verbose:1 = 进度条
    #callbacks:一系列可以在训练时使用的回调函数。
    #validation_data:用来评估损失，以及在每轮结束时的任何模型度量指标。 模型将不会在这个数据上进行训练。
    #steps_per_epoch: 整数或 None。 在声明一个轮次完成并开始下一个轮次之前的总步数（样品批次）。
    #validation_steps: 只有在指定了 steps_per_epoch时才有用。停止前要验证的总步数（批次样本）。
    new_model.fit_generator(train_gen(),
                            #steps_per_epoch=num_train_samples // batch_size,
                            steps_per_epoch=250,
                            validation_data=valid_gen(),
                            #validation_steps=num_valid_samples // batch_size,
                            validation_steps=250,
                            #epochs=epochs,
                            epochs=10,
                            verbose=1,
                            callbacks=callbacks,
                            #use_multiprocessing=True
                            #workers=get_available_cpus() // 2
                            )
