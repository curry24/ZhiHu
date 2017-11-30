# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# train_id = pd.read_csv('data/question_train_set.txt', delimiter='\t', header=None, nrows=5)
# train_new = train_id.ix[:, 2]
# train_new.to_csv('data/new_train.txt', header=None, index=None, sep='\t')





def read_test():

    word_name = '/home/niyao/zhaolei/ZhiHu/data/word_embedding.txt'
    word_embedding = Word2Vec.load_word2vec_format(word_name, binary=False)

    reader = pd.read_table('data/question_eval_set.txt', sep='\t', header=None)
    # print(reader.iloc[0:5])
    # 计算一段文本中最大词汇数
    x_text = reader.iloc[:, 2]
    max_document_length = 0
    for i, line in enumerate(x_text):
        try:
            temp = line.split(',')
            max_document_length = max(max_document_length, len(temp))
        except:
            # 其中有一行数据为空
            pass

    print 'max_document_length:', max_document_length

    j=0
    x=[]
    for features in x_text:
        xi = []
        try:
            for id in features.split(','):
                if id in word_embedding:
                    xi.append(word_embedding[id])
        except:
            xi.append(np.zeros(256))
        for i in xrange(len(xi), max_document_length):
            xi.append(np.zeros(256))
        x.append(xi)
        j += 1
        if j % 1000 == 0:
            print 'load data %d' % j

    # x = np.array(x).astype(np.float32).reshape(217360, max_document_length*256)
    x = np.array(x).astype(np.float32)
    print x.shape
    print list(x[0])
    print list(x[4])

read_test( )



