""" Class and method definition for the layers in XNOR-Net
"""
import theano
import theano.tensor.nnet
import numpy as np
import lasagne   
import theano.tensor as T
import time
from external.bnn_utils import binary_tanh_unit

def binarize_conv_filters(W):   #卷积层参数二值化
   
    # symbolic binary weight
    Wb = T.cast(T.switch(T.ge(W, 0),1,-1), theano.config.floatX)    #Wb = sign(W)  shape = (filter_number,channel,height,weight)

    # weight scaling factor  
    alpha = T.mean( T.reshape(T.abs_(W), (W.shape[0], W.shape[1]*W.shape[2]*W.shape[3])), axis=1)  # alpha.shape = (filter_number,)

    return Wb, alpha #每个卷积核计算一个alpha

def binarize_conv_input(conv_input, k):   #卷积层输入二值化

    # This is from BinaryNet.
    # This acts like sign function during forward pass. and like hard_tanh during back propagation
    bin_conv_out = binary_tanh_unit(conv_input)   #???????????????????????

    # scaling factor for the activation.
    A =T.abs_(conv_input)

    # K will have scaling matrixces for each input in the batch.
    # K's shape = (batch_size, 1, map_height, map_width)
    k_shape = k.eval().shape    #???????????????
    pad = (k_shape[-2]/2, k_shape[-1]/2)   # k(小写)的height和weight应为奇数，设置这样的padding可以保证K(大写) = A*k(小写) 的形状与A一致
    # support the kernel stride. This is necessary for AlexNet
    K = theano.tensor.nnet.conv2d(A, k, border_mode=pad)

    return bin_conv_out, K  #返回的是输入的二值化结果 和 K(upper case)    K里有各个输入单元的beta
    


def binarize_fc_weights(W):  #全连接层参数二值化
    # symbolic binary weight
    Wb = T.cast(T.switch(T.ge(W, 0),1,-1), theano.config.floatX)   #Wb=sign(W)

    alpha = T.mean(T.abs_(W), axis=0) #是个秩    注意W.shape = (X[L],X[L+1]) 每列是一个神经元的参数向量，所有axis=0
    return Wb, alpha

def binarize_fc_input(fc_input):  #全连接层输入二值化

    bin_out = binary_tanh_unit(fc_input)
    
    if(fc_input.ndim == 4):  # prev layer is conv or pooling. hence compute the l1 norm using all maps
        beta = T.mean(T.abs_(fc_input), axis=[1, 2, 3])

    else: # feeding layer is FC layer
        beta = T.mean(T.abs_(fc_input), axis=1)

    return bin_out, beta     #   beta.shape = (batch_num,)



class Conv2DLayer(lasagne.layers.Conv2DLayer):  #定义卷积操作
    """ Binary convolution layer which performs convolution using XNOR and popcount operations.
    This is followed by the scaling with input and weight scaling factors K and alpha respectively.
    """

    def __init__(self, incoming, num_filters, filter_size, xnor=True, **kwargs):

        """
        Parameters
        -----------
        incoming : layer or tuple 输入数据的shape   （或者前一层层对象） 
        num_filters: int
        filter_size: (height,weight)
        """
        self.xnor = xnor

        
        no_inputs = incoming.output_shape[1]   # 输入数据的channel
        shape = (num_filters, no_inputs, filter_size[0], filter_size[1])  #卷积核的shape

        num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])    #单个卷积核的参数个数
        num_units = int(np.prod(filter_size)*num_filters)        #？？？？？？
        self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))   ###？？？？？？

        if(self.xnor): #二值化卷积
            #权重 -1~1 均匀分布初始化
            super(Conv2DLayer, self).__init__(incoming,num_filters, filter_size, W=lasagne.init.Uniform((-1, 1)), **kwargs)
            self.params[self.W] = set(['xnor']) #？？？？                #set是一个无序且不重复的元素集合
        else:#普通卷积
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs) #参数按照一般情况自动初始化


        if self.xnor:
            # 卷积核求平均计算二值化因子
            #beta_filter.shape = (num of filters,channel,height,width) 值均为 1/(chanel*height*width)
            beta_filter = np.ones(shape=shape).astype(np.float32) / (no_inputs*filter_size[0]*filter_size[1])  #shape是line 80 的
            self.beta_filter = self.add_param(beta_filter, shape, name='beta_filter', trainable=False, regularizable=False) # shape在line 80
            Wb = np.zeros(shape=self.W.shape.eval(), dtype=np.float32)
            #alpha = np.ones(shape=(num_filters,), dtype=np.float32)
            xalpha = lasagne.init.Constant(0.1)
            
            self.xalpha = self.add_param(xalpha, [num_filters,], name='xalpha', trainable=False, regularizable=False)
            #self.B = self.add_param(Wb, Wb.shape, name='B', trainable=False, regularizable=False)
            #print self.Wb

    def convolve(self, input, deterministic=False, **kwargs):
        """ Binary convolution. Both inputs and weights are binary (+1 or -1)
        This overrides convolve operation from Conv2DLayer implementation
        """
        if(self.xnor):
            # compute the binary inputs H and the scaling matrix K
            input, K = binarize_conv_input(input, self.beta_filter)

            # Compute the binarized filters are the scaling matrix
            self.Wb, alpha = binarize_conv_filters(self.W)
            if not deterministic:
                old_alpha = theano.clone(self.xalpha, share_inputs=False)
                old_alpha.default_update = alpha
                alpha += 0*old_alpha
            else:
                alpha = self.xalpha 

            # TODO: Use XNOR ops for the convolution. As of now using Lasagne's convolution for
            # functionality verification.
            # approx weight tensor
            #W_full_precision = self.Wb * alpha.dimshuffle(0, 'x', 'x', 'x')
            Wr = self.W

            self.W = self.Wb

            feat_maps = super(Conv2DLayer, self).convolve(input, **kwargs)
            # restore the approx full precision weight for gradiant computation
            #self.W = W_full_precision
            self.W = Wr

            # scale by K and alpha
            # FIXME: Actually we are scaling after adding bias here. Need to scale first and then add bias.
            # The super class method automatically adds bias. Somehow need to overcome this..
            # may subtract the bias, scale by alpha and beta ans then add bias ?
            feat_maps = feat_maps * K

            feat_maps = feat_maps * alpha.dimshuffle('x', 0, 'x', 'x')
        else:
            feat_maps = super(Conv2DLayer, self).convolve(input, **kwargs)
    
        return feat_maps

class DenseLayer(lasagne.layers.DenseLayer):
    """Binary version of fully connected layer. XNOR and bitcount ops are used for 
    this in a similar fashion as that of Conv Layer.
    """

    def __init__(self, incoming, num_units, xnor=True, **kwargs):
        """ XNOR-Net fully connected layer
        """
        self.xnor = xnor
        num_inputs = int(np.prod(incoming.output_shape[1:]))
        self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))
        if(self.xnor):
            super(DenseLayer, self).__init__(incoming, num_units,  W=lasagne.init.Uniform((-1, 1)), **kwargs)
            self.params[self.W]=set(['xnor'])
        else:
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)

        if self.xnor:
            #Wb = np.zeros(shape=self.W.shape.eval(), dtype=np.float32)
            xalpha = np.zeros(shape=(num_units,), dtype=np.float32)
            self.xalpha = self.add_param(xalpha, xalpha.shape, name='xalpha', trainable=False, regularizable=False)
            #self.Wb = self.add_param(Wb, Wb.shape, name='Wb', trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        """ Binary dense layer dot product computation
        """
        if(self.xnor):
            # binarize the input
            bin_input, beta = binarize_fc_input(input)

            # compute weight scaling factor.
            self.Wb, alpha = binarize_fc_weights(self.W)
            if not deterministic:
                old_alpha = theano.clone(self.xalpha, share_inputs=False)
                old_alpha.default_update = alpha
                alpha += 0*old_alpha
            else:
                alpha = self.xalpha

            #W_full_precision = self.Wb * alpha.dimshuffle('x', 0)
            Wr = self.W
            self.W = self.Wb
                
            fc_out = super(DenseLayer, self).get_output_for(bin_input, **kwargs)
            # scale the output by alpha and beta
            # FIXME: Actually we are scaling after adding bias here. Need to scale first and then add bias.
            # The super class method automatically adds bias. Somehow need to overcome this..
            # may subtract the bias, scale by alpha and beta ans then add bias ?
            fc_out = fc_out * beta.dimshuffle(0, 'x')

            fc_out = fc_out * alpha.dimshuffle('x', 0)
            
            #self.W = W_full_precision
            self.W = Wr
        else:
            fc_out = super(DenseLayer, self).get_output_for(input, **kwargs)

        return fc_out

        # find the dot product
        # scale the output by alpha and beta

