# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec

# test_id = pd.read_csv('data/question_eval_set.txt', delimiter='\t', header=None)
# test_pd = test_id.ix[:, 2]
# for i in xrange(11):
#     test_name = 'data/test_block_'+str(i)+'.txt'
#     if (i + 1) * 20000 < test_pd.shape[0]:
#         test_pd.iloc[i * 20000:(i + 1) * 20000].to_csv(
#             test_name, header=None, index=None, sep='\t')
#     else:
#         test_pd.iloc[i * 20000:test_pd.shape[0]].to_csv(
#             test_name, header=None, index=None, sep='\t')
def write_matrix(mat, file_stream, max_document_length, vec_size=256):
    line = ''
    for i in xrange(len(mat)):
        for value in mat[i]:
            line += ' {0}'.format(value)
    for i in xrange(len(mat), max_document_length):
        for j in xrange(vec_size):
            line += ' 0'
    file_stream.write(line + '\n')


def generate_cnn_vec(word_embedding, documents, file_name):
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
            if id in word_embedding:
                mat.append(word_embedding[id])
        write_matrix(mat, cnn_file, max_document_length)
        # print len(mat)
        if i % 1000== 0:
            print '%d, genera %f' % (i, float(i) / length)
    cnn_file.close()



test_name = 'data/test_block_0.txt'
test_id = pd.read_table('data/test_block_0.txt', header=None, nrows=1)
# print test_id
test_complex = np.array(test_id)
# print test_complex[0:5]
test = []
for i in test_complex:
    sentence = re.sub('[^a-zA-Z0-9]', ' ', str(i)).split()
    test.append(sentence)
print test

b= ['w11','w54']
print b
test_b= []
test_b.append(b)
print test_b

word_name = '/home/niyao/zhaolei/ZhiHu/data/word_embedding.txt'
word_embedding = Word2Vec.load_word2vec_format(word_name, binary=False)
test_vec_file = 'test_a.txt'
# generate_cnn_vec(word_embedding, a, test_vec_file)
generate_cnn_vec(word_embedding, test_b, 'test_b.txt')
