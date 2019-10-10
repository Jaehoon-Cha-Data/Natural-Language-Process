# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:02:00 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
import tensorflow as tf
import numpy as np

def cycle_fn(iteration, base_lr, max_lr, stepsize):
    cycle = np.floor(1+iteration/(2*stepsize))
    x = np.abs(iteration/stepsize - 2*cycle +1)
    lr = base_lr + (max_lr - base_lr)*np.maximum(0, (1-x))
    return np.float32(lr)


def cycle_lr(base_lr, max_lr, iter_in_batch, epoch_for_cycle, ratio, total_epochs):
    iteration = 0;
    Lr = [];
    stepsize = (iter_in_batch*epoch_for_cycle)/2.
    for i in range(total_epochs):
        for j in range(iter_in_batch):
            Lr.append(cycle_fn(iteration, base_lr = base_lr, 
                            max_lr = max_lr, stepsize = stepsize))
            iteration+=1
    final_iter = np.int((total_epochs/epoch_for_cycle)*stepsize*2*ratio)
    Lr = np.array(Lr)
    Lr[final_iter:] = base_lr*0.01
    return Lr

def step_lr(lr_set, iter_in_epoch, total_epochs):
    Lr = []
    n_lr = len(lr_set)
    for i in range(n_lr - 1):
        sub_lr = [lr_set[i]] * int(total_epochs/n_lr)*iter_in_epoch
        Lr += sub_lr
    sub_lr = [lr_set[-1]]* int(total_epochs/n_lr)*iter_in_epoch*2
    Lr += sub_lr
    Lr = np.array(Lr)
    return Lr

def Batch_norm(x, training, name = None, reuse = None):
    return tf.contrib.layers.batch_norm(inputs=x,
                                        decay=0.997, epsilon=1e-5,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=training, fused=True, reuse = reuse,
                                        scope = name)
   
