# -*- coding:utf-8 -*-
from utils import *
import random
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import re


def load_train( ):
    train_id = pd.read_csv('data/train_example.txt', delimiter='\t', names=['0', '1', '2', '3', '4'])
    # print train_id
    train_pd = train_id.drop('0', axis=1)
    # print train_pd
    train_complex = np.array(train_pd)
    train = []
    for i in train_complex:
        sentence = re.sub('[^a-zA-Z0-9]', ' ', str(i)).split()
        train.append(sentence)

    # label_new = pd.read_csv('data/label_new_example.csv', header=None)
    # # print label_new
    # label = np.array(label_new).tolist()
    # # print label

    label_id = pd.read_csv('data/label_example.txt', delimiter='\t', names=['0', '1'])
    label_pd = label_id.drop('0', axis=1)
    label = np.array(label_pd)
    label_string = []
    for i in label:
        item = re.sub('[^-?\d+$]', ' ', str(i)).split()
        label_string.append(item)

    label_one = []  #选择每个question的第一个topic作为label，因此是one label
    for i in range(len(label_string)):
        for j in range(len(label_string[i])):
            if j == 0:
                label_one.append(label_string[i][j])

    return train, label_one

def write_matrix(mat, label, file_stream, max_document_length, vec_size=256):
    line = str(label)
    for i in xrange(len(mat)):
        for value in mat[i]:
            line += ' {0}'.format(value)
    for i in xrange(len(mat), max_document_length):
        for j in xrange(vec_size):
            line += ' 0'
    file_stream.write(line + '\n')



def generate_cnn_vec(char_embedding, word_embedding, documents, labels, file_name):
    cnn_file = open(file_name, 'w')
    length = len(labels)
    max_document_length = 0
    for i in xrange(length):
        doc_length = len(documents[i])
        max_document_length = max(max_document_length, doc_length)

    for i in xrange(length):
        mat = []
        # 判断的是id
        for id in documents[i]:
            if id in char_embedding:
                mat.append(char_embedding[id])
            # elif id in word_embedding:
            #     mat.append(word_embedding[id])
            # else:
            #     mat.append(np.zeros(256))
            if id in word_embedding:
                mat.append(word_embedding[id])
        write_matrix(mat, labels[i], cnn_file, max_document_length)
        print len(mat)
        if i % 5== 0:
            print '%d, genera %f' % (i, float(i) / length)
    cnn_file.close()


def generate_cnn_train_test(char_name, word_name):
    cnn_vec_dir = 'cnn_vec'
    ensure_path(cnn_vec_dir)

    char_embedding = Word2Vec.load_word2vec_format(char_name, binary=False)
    word_embedding = Word2Vec.load_word2vec_format(word_name, binary=False)
    train_doc, train_label = load_train()
    # test_doc, test_label = load_test()
    train_vec_file = cnn_vec_dir + '/' + 'train.txt'
    # test_vec_file = cnn_vec_dir + '/' + 'test.txt'
    generate_cnn_vec(char_embedding, word_embedding, train_doc, train_label, train_vec_file)
    print 'generate cnn train feature ok'
    # generate_cnn_vec(char_embedding, word_embedding, test_doc, test_label, test_vec_file)
    # print 'generate cnn test feature ok'

# char_name = 'D:\BaiduNetdiskDownload\ieee_zhihu_cup\char_embedding.txt'
# word_name = 'D:\BaiduNetdiskDownload\ieee_zhihu_cup\word_embedding.txt'

char_name = '/home/niyao/zhaolei/ZhiHu/data/char_embedding.txt'
word_name = '/home/niyao/zhaolei/ZhiHu/data/word_embedding.txt'


generate_cnn_train_test(char_name, word_name)
