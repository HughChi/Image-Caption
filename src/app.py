# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')
# 引入必要的包
import os
import pickle
import random

import cv2 as cv
import keras.backend as K

from config import test_a_image_folder, img_rows, img_cols, best_model
from forward import build_model
from generated import test_gen

if __name__ == '__main__':
    channel = 3
    #读取权重文件
    model_weights_path = os.path.join('models', best_model)
	#创建模型
    model = build_model()
	#加载权重
    model.load_weights(model_weights_path)
    #加载语料库
    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
	
	#进行处理  便于模型读取文字数据和预测
    idx2word = sorted(vocab)
    word2idx = dict(zip(idx2word, range(len(vocab))))
    
    print(model.summary())

    test_gen()

    #读取图片
    encoded_test_a = pickle.load(open('data/encoded_test_a_images.p', 'rb'))
   
    names = [f for f in encoded_test_a.keys()]
     
    samples = names

    sentences = []
    for i in range(len(samples)):
	    #依次取图片
        image_name = samples[i]
        filename = os.path.join(test_a_image_folder, image_name)
        # # print('Start processing image: {}'.format(filename))
        # image_input = np.zeros((1, 2048))
        # image_input[0] = encoded_test_a[image_name]
        #
        # start_words = [start_word]
        # while True:
        #     text_input = [word2idx[i] for i in start_words]
        #     text_input = sequence.pad_sequences([text_input], maxlen=max_token_length, padding='post')
        #     preds = model.predict([image_input, text_input])
        #     # print('output.shape: ' + str(output.shape))
        #     word_pred = idx2word[np.argmax(preds[0])]
        #     start_words.append(word_pred)
        #     if word_pred == stop_word or len(start_word) > max_token_length:
        #         break
        #使用beam_search机制进行预测
        from beam_search import beam_search_predictions
        
        candidate = beam_search_predictions(model, image_name, word2idx, idx2word, encoded_test_a,
                                            beam_index=20)
        #打印结果
        print(candidate)
        sentences.append(candidate)
        #读取图片 并调整其大小
        img = cv.imread(filename)
        #img = cv.resize(img, (img_rows, img_cols), cv.INTER_CUBIC)
        if not os.path.exists('images'):
            os.makedirs('images')
        cv.imwrite('images/{}_image.jpg'.format(i), img)
    #将预测产生的描述信息输出到demo.txt文件中
    with open('demo.txt', 'w') as file:
        file.write('\n'.join(sentences))

    K.clear_session()