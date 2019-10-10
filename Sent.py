# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:09:46 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
import numpy as np



class Sent(object):
    def __init__(self, dataDir, seq_len):
        self.dataDir = dataDir
        self.seq_len = seq_len
        
        self.timetxt = []
        with open(dataDir, 'r', encoding = 'utf-8') as f:
            for ele in f:
                ele = ele.replace('\t', ' ').replace('\n', '').replace('.', '').replace("'", '').replace(",", '').lower()
                self.timetxt.append(ele)

        self.timetxt = self.timetxt[:10]
        self.sent = '. '.join(self.timetxt)
        self.char_set = list(set(self.sent))
        self.char_dic = {w: i for i, w in enumerate(self.char_set)}
        self.idx_dic = {i: w for i, w in enumerate(self.char_set)}
        
        self.num_char = len(self.char_set)

        self.make_sequece()
        
        self.num_examples = len(self.train_x)
        
        self.index_in_epoch = 0
        
    def make_sequece(self):
        self.X = []
        self.Y = []
        for i in range(0, len(self.sent) - self.seq_len):
            x_str = self.sent[i:i + self.seq_len]
            y_str = self.sent[i + 1: i + self.seq_len + 1]
        
            x = [self.char_dic[c] for c in x_str]  # x str to index
            y = [self.char_dic[c] for c in y_str]  # y str to index
        
            self.X.append(x)
            self.Y.append(y)
            
        self.train_x = np.array(self.X)
        self.train_y = np.array(self.Y)        
        
        
    def next_train_batch(self, batch_size, shuffle=True):
        start = self.index_in_epoch
        if start == 0:
            perm0 = np.arange(self.num_examples)
            np.random.shuffle(perm0)
            self._train_x = self.train_x[perm0]
            self._train_y = self.train_y[perm0]
        if start + batch_size > self.num_examples:
            rand_index = np.random.choice(self.num_examples, size = (batch_size), replace = False)
            epoch_x, epoch_y = self.train_x[rand_index], self.train_y[rand_index]
            self.index_in_epoch = 0
            return epoch_x, epoch_y
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            epoch_x, epoch_y = self._train_x[start:end], self._train_y[start:end]
            return epoch_x, epoch_y
                    


    
