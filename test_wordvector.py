import pandas as pd
import numpy as np

# test_id = pd.read_csv('data/question_eval_set.txt', delimiter='\t', header=None)
# test_new = test_id.ix[:, 2]
# test_new.to_csv('data/new_test.txt', header=None, index=None, sep='\t')
x = []
x0 = ['w1', 'w2']
x1 = ['w3', 'w4', 'w5']
x2 = []
x.append(x0)
x.append(x1)
x.append(x2)
print x

for i in xrange(len(x)):
    mat = []
    for id in x[i]:
        mat.append(id)
    print len(mat)
    print mat


