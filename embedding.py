#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 19:36:22 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt


## call words
timetxt = []
with open('time.txt', 'r', encoding = 'utf-8') as f:
    for ele in f:
        ele = ele.replace('\t', ' ').replace('\n', '').replace('.', '').replace("'", '').replace(",", '').lower()
        timetxt.append(ele)


## remove redundant words
def Redundant_words(txt):
    new_txt = []
    for t in txt:
        new_t = []
        for w in t.split():
            if w not in ['a', 'is', 'be', 'the', 'to', 'for', 'we', 'i', 'and',
                         'but', 'it', 'it’s', 'you', 'you’re', 'you’ve', 'your',
                         'are','that','what', 'when','who','or','our','of','on']:
                new_t.append(w)
        new_txt.append(' '.join(new_t))
    return new_txt
                
red_timetxt = Redundant_words(timetxt)



### find unique words
all_words = []
for sent in red_timetxt:
    sent_words = sent.split()
    for word in sent_words:
        all_words.append(word)
            
n_all_words = len(all_words)

### find unique words
unique_words = np.unique(all_words)
n_words = len(unique_words)


#### change char to index
char2idx = {c: i for i, c in enumerate(unique_words)}  # char -> index
count_words = {c: len(np.where(c == np.array(all_words))[0])
                 for i, c in enumerate(unique_words)} 


### make data inputs and outputs
in_out = []
window_size = 2
for sent in red_timetxt:
    sent = sent.split()
    for i, w in enumerate(sent):
        neighbors = sent[max(0,i-window_size):min(i+window_size+1,len(sent))].copy()
        for n in neighbors:
            if n != w:
                in_out.append([w, n])
        
        
### make dataset
df = pd.DataFrame(in_out, columns = ['input', 'output'])

        
### word to onehot
eye = np.eye(n_words)
def Word2onehot(char):
    return eye[char2idx[char]]


### to one hot 
train_x = []
train_y = []
for inp, outp in zip(df['input'], df['output']):
    train_x.append(Word2onehot(inp))
    train_y.append(Word2onehot(outp))

train_x = np.array(train_x)
train_y = np.array(train_y)


embdd_dim = 2

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)


### make model graph
x = tf.placeholder(tf.float32, shape=(None, n_words))
y = tf.placeholder(tf.float32, shape=(None, n_words))

W1 = tf.Variable(xavier_init(n_words, embdd_dim))
b1 = tf.Variable(tf.zeros(embdd_dim)) 
W2 = tf.Variable(xavier_init(embdd_dim, n_words))
b2 = tf.Variable(tf.zeros(n_words))

hidden_layer = tf.add(tf.matmul(x,W1), b1)
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))

loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(y ,tf.log(prediction)), 1))
optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(loss)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer()) 

iteration = 100000
for i in range(iteration):
    _, l = sess.run([optimizer, loss], feed_dict={x: train_x, y: train_y})
    if i % 3000 == 0:
        print('iteration '+str(i)+' loss is : ', l)
        

### to embdd vector
word2vect = np.array([Word2onehot(w) for w in unique_words])
embeddings = sess.run(W1 + b1, feed_dict={x:word2vect})
print(embeddings)

embdd_df = pd.DataFrame(embeddings, columns = ['x1', 'x2'])
embdd_df['word'] = unique_words
embdd_df = embdd_df[['word', 'x1', 'x2']]


fig, ax = plt.subplots(figsize = (10,10))

for word, x1, x2 in zip(embdd_df['word'], embdd_df['x1'], embdd_df['x2']):
    if word != 'time':
        ax.annotate(word, (x1,x2 ), size = count_words[word]*5)
    else:
        ax.annotate(word, (x1,x2 ), size = count_words[word]*2)

x_min = np.min(embeddings, axis=0)[0] - 1.0
y_min = np.min(embeddings, axis=0)[1] - 1.0
x_max = np.max(embeddings, axis=0)[0] + 1.0
y_max = np.max(embeddings, axis=0)[1] + 1.0
 
plt.xlim(x_min,x_max)
plt.ylim(x_min,y_max)
plt.show()
