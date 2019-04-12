# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')
# import the necessary packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pickle
import random

import cv2 as cv
import keras.backend as K
import numpy as np
from keras.preprocessing import sequence
#从config文件中引入一些参数 包括token最大长度 测试文件夹长度 最优的模型参数
from config import max_token_length, test_a_image_folder, best_model
from forward import build_model
from generated import test_gen

#使用训练好的模型对图片进行测试
def beam_search_predictions(model, image_name, word2idx, idx2word, encoding_test, beam_index=3):
    start = [word2idx["<start>"]]
    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_token_length:
        temp = []
        for s in start_word:
		    #对序列进行填充的预处理，在其后添0，使其序列统一大小为max_token_length
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_token_length, padding='post')
			#每次取一个图片进行测试
            e = encoding_test[image_name]
			#使用模型对该图片进行测试
            preds = model.predict([np.array([e]), np.array(par_caps)])
            #从预测的结果中取前beam_index个
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a
            # new list so as to put them via the model again
			# 创建一个新的list结构 将预测出的词和词的概率以组对的形式存储
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
        #将处理好的预测值赋值回start word
        start_word = temp
        # Sorting according to the probabilities
		# 根据概率排序
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
		#获得最有可能正确的词
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
	#根据id取出单词
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []
    #组合成句
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break

    final_caption = ''.join(final_caption[1:])
    return final_caption


if __name__ == '__main__':
    #图片的channel为3
    channel = 3
    #设置模型权重的地址
    model_weights_path = os.path.join('models', best_model)
    print('模型加载中...')
	#创建模型
    model = build_model()
	#加载模型权重
    model.load_weights(model_weights_path)
    print('模型加载完毕...')

    #print(model.summary())  
    #加载语料库
    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
	#将word转化为数字  方便输入网络 进行预测
    idx2word = sorted(vocab)
    word2idx = dict(zip(idx2word, range(len(vocab))))
    print('语料库加载完毕...')

    test_gen()

    #加载测试图片
    encoding_test = pickle.load(open('data/encoded_test_a_images.p', 'rb'))
    #随机取测试图片
    names = [f for f in encoding_test.keys()]
    samples = names
    #samples = random.sample(names, 1)
    sentences = []

    for i in range(len(samples)):
        image_name = samples[i]

        image_input = np.zeros((1, 2048))
        image_input[0] = encoding_test[image_name]
        #获取图片的名称
        filename = os.path.join(test_a_image_folder, image_name)
        # print('Start processing image: {}'.format(filename))
        #设置不同的预测参数，并放到beam_search_predictions中进行预测
        print('描述的图片为:',image_name)

        candidate1=beam_search_predictions(model, image_name, word2idx, idx2word, encoding_test, beam_index=1)
        print('Beam Search, k=1:',candidate1)
        sentences.append(candidate1)

        candidate2=beam_search_predictions(model, image_name, word2idx, idx2word, encoding_test, beam_index=3)
        print('Beam Search, k=3:',candidate2)
        sentences.append(candidate2)

        candidate3=beam_search_predictions(model, image_name, word2idx, idx2word, encoding_test, beam_index=5)
        print('Beam Search, k=5:',candidate3)
        sentences.append(candidate3)

        candidate4=beam_search_predictions(model, image_name, word2idx, idx2word, encoding_test, beam_index=7)
        print('Beam Search, k=7:',candidate4)
        sentences.append(candidate4)

        #读取图片
        img = cv.imread(filename)
		#resise到固定大小 256*256
        #img = cv.resize(img, (256, 256), cv.INTER_CUBIC)
        if not os.path.exists('images'):
            os.makedirs('images')
		#将修改后的图片重新写回
        cv.imwrite('images/{}_bs_image.jpg'.format(i), img)

    #将预测产生的描述信息输出到demo.txt文件中
    with open('demo.txt', 'w') as file:
        file.write('\n'.join(sentences))

    K.clear_session()