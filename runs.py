# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:05:49 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
import numpy as np
import tensorflow as tf
from models import LSTM
import argparse
from collections import OrderedDict
from Sent import Sent
import os
from networks import step_lr


def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Error')
        
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 'LSTM')
    parser.add_argument('--datasets', type = str, default = 'time')
    parser.add_argument('--epochs', type = int, default = 500)
    parser.add_argument('--batch_size', type = int, default =128)
    parser.add_argument('--lr_sets', type = list, default = [0.1,
                                                                    0.05,
                                                                    0.01])
    parser.add_argument('--cycle_epoch', type = int, default = 20)
    parser.add_argument('--cycle_ratio', type = float, default = 0.7)
    parser.add_argument('--z_dim', type = int, default = 128)
    parser.add_argument('--seq_len', type = int, default = 10)
    parser.add_argument('--num_hid_layers', type = int, default = 2)
    
    args = parser.parse_args()
    
    config = OrderedDict([
            ('model_name', args.model_name),
            ('datasets', args.datasets),
            ('epochs', args.epochs),
            ('batch_size', args.batch_size),
            ('lr_sets', args.lr_sets),
            ('cycle_epoch', args.cycle_epoch),
            ('cycle_ratio', args.cycle_ratio),
            ('z_dim', args.z_dim),
            ('seq_len', args.seq_len),
            ('num_hid_layers', args.num_hid_layers)])
    
    return config
    
config = parse_args()


### call data ###
sent = Sent('time.txt', config['seq_len'])
n_samples = sent.num_examples 


### call models ###
model = LSTM(sent.num_char, config['num_hid_layers'])


### make folder ###
mother_folder = config['model_name']
try:
    os.mkdir(mother_folder)
except OSError:
    pass    


### outputs ###
pred, loss = model.Forward()


### cyclic learning rate ###
iter_per_epoch = int(n_samples/config['batch_size']) 

Lr = step_lr(config['lr_sets'], iter_per_epoch, config['epochs'])

cy_lr = tf.placeholder(tf.float32, shape=(),  name = "cy_lr")

folder_name = os.path.join(mother_folder, config['model_name']+'_'+config['datasets'])


### run ###
with tf.Session() as sess:   
    optimizer = tf.train.AdamOptimizer(learning_rate=cy_lr).minimize(loss)
    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    model_save_name = os.path.join(folder_name, config['model_name']+'.ckpt')
    
    try:
        os.mkdir(folder_name)
    except OSError:
        pass    
    
    iteration = 0
    for epoch in range(config['epochs']):
        epoch_loss = 0
        for iter_in_epoch in range(iter_per_epoch):
            epoch_x, epoch_y = sent.next_train_batch(config['batch_size'])
            _, c = sess.run([optimizer, loss], feed_dict = {model.x: epoch_x, model.y: epoch_y,
                                           cy_lr: Lr[iteration]})
            epoch_loss += c
            iteration+=1
            
        print('Epoch', epoch, 
              'completed out of', config['epochs'], 'loss:', epoch_loss/int(iter_per_epoch))
        
        
    pred_list =[]
    for i in range(sent.num_examples):
        p_out = sess.run(pred, feed_dict = {model.x: sent.train_x[i].reshape(1,-1)})
        p_onehot_last = np.argmax(p_out, -1)[0][-1]
        pred_list.append(p_onehot_last)
    p_sentences = [sent.idx_dic[s] for s in pred_list]
    p_sentences = ''.join(p_sentences)
    
    print("----------X-----------")
    print(sent.sent[:-sent.seq_len])
    print("----------Y-----------")
    print(sent.sent[sent.seq_len:])
    print("------prediction-------")
    print(p_sentences)
     
            







