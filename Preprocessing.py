# -*- coding:utf-8 -*-
from utils import *
import random
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import re


def load_train( ):
    train_id = pd.read_csv('data/question_train_set.txt', delimiter='\t', header=None, nrows=10000)
    # print train_id
    # train_pd = train_id.drop('0', axis=1)
    train_pd = train_id.ix[:, 2]
    # print train_pd
    train_complex = np.array(train_pd)
    train = []
    for i in train_complex:
        sentence = re.sub('[^a-zA-Z0-9]', ' ', str(i)).split()
        train.append(sentence)

    # label_reader = pd.read_table('data/topic_info.txt', sep='\t', header=None)
    # labels = list(label_reader.iloc[:, 0])
    # my_labels = []
    # for label in labels:
    #     my_labels.append(label)
    # # 建立topic字典
    # topic_dict = {}
    # for i, label in enumerate(my_labels):
    #     topic_dict[label] = i

    # label_input = pd.read_csv('data/question_topic_train_set.txt', delimiter='\t', names=['0', '1'], nrows=10)
    # y = []
    # for i in xrange(label_input.shape[0]):
    #     # 根据“,”切分话题id
    #     temp_topic = label_input.iloc[i][1].split(',')
    #     if (len(temp_topic) > 5):
    #         temp_topic = temp_topic[0:5]
    #     label = np.zeros(1999)
    #     for topic in temp_topic:
    #         # 判断该label是否在label文件中，并得到该行
    #         label_num = topic_dict[int(topic)]
    #         label[int(label_num)] = 1
    #     y.append(list(label))
    # print y[0]
    # print y[3]
    # print y[4]
    return train



def load_test(test_name):
    test_id = pd.read_table(test_name, header=None)
    # print train_id
    # train_pd = train_id.drop('0', axis=1)
    # test_pd = test_id.ix[:, 2]
    # print train_pd
    test_complex = np.array(test_id)
    test = []
    for i in test_complex:
        sentence = re.sub('[^a-zA-Z0-9]', ' ', str(i)).split()
        test.append(sentence)

    return test

def write_matrix(mat, file_stream, max_document_length, vec_size=256):
    line = ''
    for i in xrange(len(mat)):
        for value in mat[i]:
            line += ' {0}'.format(value)
    for i in xrange(len(mat), 76):
        for j in xrange(vec_size):
            line += ' 0'
    file_stream.write(line + '\n')



def generate_cnn_vec(char_embedding, word_embedding, documents, file_name):
    cnn_file = open(file_name, 'w')
    length = len(documents)
    max_document_length = 0
    for i in xrange(length):
        doc_length = len(documents[i])
        max_document_length = max(max_document_length, doc_length)

    print 'max document length of '+ file_name+' is %d' % max_document_length

    for i in xrange(length):
        mat = []
        # 判断的是id
        for id in documents[i]:
            # if id in char_embedding:
            #     mat.append(char_embedding[id])
            # elif id in word_embedding:
            #     mat.append(word_embedding[id])
            # else:
            #     mat.append(np.zeros(256))
            if id in word_embedding:
                mat.append(word_embedding[id])
        write_matrix(mat, cnn_file, max_document_length)
        # print len(mat)
        if i % 1000== 0:
            print '%d, genera %f' % (i, float(i) / length)
    cnn_file.close()


def generate_cnn_train_test(char_name, word_name):
    cnn_vec_dir = 'cnn_vec'
    ensure_path(cnn_vec_dir)

    char_embedding = Word2Vec.load_word2vec_format(char_name, binary=False)
    word_embedding = Word2Vec.load_word2vec_format(word_name, binary=False)
    # train_doc = load_train()
    # train_vec_file = cnn_vec_dir + '/' + 'train.txt'
    # generate_cnn_vec(char_embedding, word_embedding, train_doc, train_vec_file)
    # print 'generate cnn train feature ok'
    for i in xrange(11):
        test_name = 'data/test_block_'+str(i)+'.txt'
        test_doc = load_test(test_name)
        test_vec_file = cnn_vec_dir + '/' + 'test_'+str(i)+'.txt'
        generate_cnn_vec(char_embedding, word_embedding, test_doc, test_vec_file)
    print 'generate cnn test feature ok'

# char_name = 'D:\BaiduNetdiskDownload\ieee_zhihu_cup\char_embedding.txt'
# word_name = 'D:\BaiduNetdiskDownload\ieee_zhihu_cup\word_embedding.txt'

char_name = '/home/niyao/zhaolei/ZhiHu/data/char_embedding.txt'
word_name = '/home/niyao/zhaolei/ZhiHu/data/word_embedding.txt'


generate_cnn_train_test(char_name, word_name)
