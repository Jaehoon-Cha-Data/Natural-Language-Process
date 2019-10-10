# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:03:45 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
import tensorflow as tf

class LSTM(object):
    def __init__(self, num_char, num_layers):
        self.num_char = num_char
        self.num_layers = num_layers
        
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])        
        self.lstm_cell()
        self.stack_lstm()
        
    def lstm_cell(self):
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.num_char, state_is_tuple=True)
           
    def stack_lstm(self):
        self.multi_cells = tf.contrib.rnn.MultiRNNCell([self.cell for _ in range(self.num_layers)], state_is_tuple=True)

    def Forward(self):
        self.x_onehot = tf.one_hot(self.x, self.num_char)
        self.hidden_outputs, self.states = tf.nn.dynamic_rnn(self.multi_cells, self.x_onehot, dtype=tf.float32)      
        
        self.outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.num_char, activation=None))(self.hidden_outputs)
        
        self.weights = tf.ones([tf.shape(self.x)[0], tf.shape(self.x)[1]])
        
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs, targets=self.y, weights=self.weights)
        
        def Summaray():
            tf.summary.scalar('loss', self.loss)
        
        Summaray()
        
        return self.outputs, self.loss
    
    
    
    