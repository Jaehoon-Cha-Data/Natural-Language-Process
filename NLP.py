# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:21:48 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

from https://www.awakenthegreatnesswithin.com/35-inspirational-quotes-time/
"""
import numpy as np
from scipy.spatial.distance import cosine

## call words
fromtxt = []
with open('time.txt', 'r', encoding = 'utf-8') as f:
    for ele in f:
        ele = ele.replace('\t', ' ').replace('\n', '').replace('.', '').replace("'", '').replace(",", '').lower()
        fromtxt.append(ele)

all_words = []
with open('time.txt', 'r', encoding = 'utf-8') as f:
    for ele in f:
        ele = ele.replace('\t', ' ').replace('\n', '').replace('.', '').replace("'", '').replace(",", '').lower()
        ele_words = ele.split()
        for word in ele_words:
            all_words.append(word)
            
n_all_words = len(all_words)

### find unique words
unique_words = np.unique(all_words)
n_words = len(unique_words)

count_words = {c: len(np.where(c == np.array(all_words))[0])
                 for i, c in enumerate(unique_words)} 
#### change char to index to one hot
char2idx = {c: i for i, c in enumerate(unique_words)}  # char -> index
eye = np.eye(n_words)
def word_one_hot(char):
    return eye[char2idx[char]]

#### make each sentence


#### sent to char
tokens_dic = {}
for i in range(len(fromtxt)):
    if len(fromtxt[i]) != 0:
        tokens_dic[i] = fromtxt[i].split()
#       
#### sent to coding   
def sent2cod(sent):
    cod = np.zeros(n_words)
    for w in sent:
        cod += eye[char2idx[w]]    
    return cod


sents2codes = {i: sent2cod(tokens_dic[s]) for i, s in enumerate(tokens_dic)}

    
#### cal IDF
def IDF(words, sents):
    char2idf = {}
    for w in words:
        w_count_sents = 0
        for sent in sents:
            if w in sent:
                w_count_sents+=1
        char2idf[w] = np.log10(len(sents)/w_count_sents)
    return char2idf             

char2idf = IDF(unique_words, fromtxt)       

#### cal TF_IDF -> 
##drawback: worse to handle synonym, frequent words have low similarity despite them import
def TF_IDF(sent):
    cod = np.zeros(n_words)
    for w in sent:
        cod += (eye[char2idx[w]]/len(sent))*char2idf[w]    
    return cod

tf_idf = {i: TF_IDF(tokens_dic[s]) for i, s in enumerate(tokens_dic)}

k = 8
sim_s2c_with_8th = []
for i in range(len(sents2codes)):
    if i != k:
        sim_s2c_with_8th.append(cosine(sents2codes[k], sents2codes[i]))
        
print("Sentence")
print(fromtxt[k])
print("is similar to ")
print(fromtxt[np.argmin(sim_s2c_with_8th)])

sim_tf_idf_with_8th = []
for i in range(len(sents2codes)):
    if i != k:
        sim_tf_idf_with_8th.append(cosine(tf_idf[k], tf_idf[i]))
        
print("Sentence")
print(fromtxt[k])
print("is similar to ")
print(fromtxt[np.argmin(sim_tf_idf_with_8th)])



