# This code is taken from the BinaryNet implementation by Matthieu Courbariaux
# The original code can be found here https://github.com/MatthieuCourbariaux/BinaryNet
# The LICENSE for this piece of code is put under this directory
from collections import OrderedDict
import time
import numpy as np

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  #类似numpy.random.RandomState()

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

# Our own rounding function, that does not set the gradient to 0 like Theano's
class Round3(UnaryScalarOp):        ###？？？？？？？？？？？？？？？？
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()
    
    def grad(self, inputs, gout):
        (gz,) = gout
        return gz, 
        
round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)


def binary_tanh_unit(x):   #？？？？？？？？？？？？？？？？？？？？？？？？？？？
    return 2.*round3(hard_sigmoid(x))-1.
    
def binary_sigmoid_unit(x):   #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    return round3(hard_sigmoid(x))

# This function computes the gradient of the binary weights
def compute_grads(loss,network):
        
    layers = lasagne.layers.get_all_layers(network)  
    grads = []
    
    for layer in layers:
    
        params = layer.get_params(xnor=True)   #xnor-net结构的网络的各个层的参数
        if params:
            # print(params[0].name)
            grads.append(theano.grad(loss, wrt=layer.Wb))   #计算loss关于二值化的权重的梯度
                
    return grads  #梯度列表


def clipping_scaling(updates,network):  ##更新参数并裁剪
    
    layers = lasagne.layers.get_all_layers(network)
    updates = OrderedDict(updates)
    
    for layer in layers:
    
        params = layer.get_params(xnor=True)
        for param in params:
            print("W_LR_scale = "+str(layer.W_LR_scale))
            
            #W_LR_scale为learning_rate扩大倍数，updates[param]为更新后的权重，param为未更新权重
            #即先更新，然后调整更新的幅度 e.g. old=10,new=12, new=10+W_LR_scale(12-10)
            updates[param] = param + layer.W_LR_scale*(updates[param] - param) 
            
            updates[param] = T.clip(updates[param], -1.0,1.0)     

    return updates


def train(train_fn,val_fn,     ##这两个function类似神经网络的forward函数
            model,             #网络模型
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,
            X_test,y_test,
            save_path=None,
            shuffle_parts=1):
    
    
    def shuffle(X,y):  # A function which shuffles a dataset
        
        # print(len(X))
        
        chunk_size = len(X)/shuffle_parts
        shuffled_range = range(chunk_size)
        
        X_buffer = np.copy(X[0:chunk_size])
        y_buffer = np.copy(y[0:chunk_size])
        
        for k in range(shuffle_parts):
            
            np.random.shuffle(shuffled_range)

            for i in range(chunk_size):
                
                X_buffer[i] = X[k*chunk_size+shuffled_range[i]]
                y_buffer[i] = y[k*chunk_size+shuffled_range[i]]
            
            X[k*chunk_size:(k+1)*chunk_size] = X_buffer
            y[k*chunk_size:(k+1)*chunk_size] = y_buffer
        
        return X,y
        
    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X,y,LR):
        
        loss = 0
        batches = len(X)/batch_size  #总共有多少个batch
        
        for i in range(batches):
            new_loss = train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size],LR)
            loss += new_loss
            ##打印出当前是第几个batch,以及截至目前各个batch的loss的平均
            print('Train batch = {:d} of {:d}\tLoss = {:f}'.format(i, batches, float(loss)/(i+1)))
        
        loss/=batches  #最终epoch结束后各个batch的loss的平均
        
        return loss
    
    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X,y):
        
        err = 0
        loss = 0
        batches = len(X)/batch_size   
        
        for i in range(batches):
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            err += new_err  ## 各个batch的错误率累加
            loss += new_loss ## 各个batch的loss累加
            #print('Val batch = {:d}'.format(i))
        
        err = err / batches * 100  #各个batch的错误了平均
        loss /= batches            #各个batch的loss平均

        return err, loss
   
 
    assert(len(X_train) == len(y_train)), 'Train data and label dimension does not match' 
    assert(len(X_val) == len(y_val)), 'Val data and label dimension does not match' 
    assert(len(X_test) == len(y_test)), 'Test data and label dimension does not match' 
    
    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train)
    best_val_err = 100
    best_epoch = 1   #模型在哪个epoch最好
    LR = LR_start
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        
        start_time = time.time()  #每个epoch计时
        
        train_loss = train_epoch(X_train,y_train,LR)
        #print('Done train epoch')
        X_train,y_train = shuffle(X_train,y_train) #每次训练完之后打乱train 数据
        
        val_err, val_loss = val_epoch(X_val,y_val)
        #print('Done validation epoch')
        
        # test if validation error went down
        if val_err <= best_val_err:
            
            best_val_err = val_err
            best_epoch = epoch+1
            #print('Starting test epoch')
            test_err, test_loss = val_epoch(X_test,y_test)   ## val错误率小的epoch才进行test操作
            
            #val错误率低的epoch才保存模型
            if save_path is not None:  
                np.savez(save_path, *lasagne.layers.get_all_param_values(model))   #保存模型    *代表这个参数将被分解
        
        epoch_duration = time.time() - start_time   #每个epoch用时
        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation error rate:    "+str(best_val_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 
        
        # decay the LR
        LR *= LR_decay 
