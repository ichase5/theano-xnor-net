""" Class and method definition for the layers in XNOR-Net
"""
import theano
import theano.tensor.nnet
import numpy as np
import lasagne
import theano.tensor as T
import time
from external.bnn_utils import binary_tanh_unit

def binarize_conv_filters(W):
    """Binarize convolution weights and find the weight scaling factor
    W : theano tensor : convolution layer weight of dimension no_filters x no_feat_maps x h x w
    """
    # symbolic binary weight
    Wb = T.cast(T.switch(T.ge(W, 0),1,-1), theano.config.floatX)
    # BinaryNet method
    #Wb = T.cast(T.switch(T.round(hard_sigmoid(W),1,-1)), theano.config.floatX)

    # weight scaling factor
    # FIXME: directly compute the mean along axis 1,2,3 instead of reshaping    
    alpha = T.mean( T.reshape(T.abs_(W), (W.shape[0], W.shape[1]*W.shape[2]*W.shape[3])), axis=1)

    return Wb, alpha

def binarize_conv_input(conv_input, k):

    # This is from BinaryNet.
    # This acts like sign function during forward pass. and like hard_tanh during back propagation
    bin_conv_out = binary_tanh_unit(conv_input)

    # scaling factor for the activation.
    A =T.abs_(conv_input)

    # K will have scaling matrixces for each input in the batch.
    # K's shape = (batch_size, 1, map_height, map_width)
    k_shape = k.eval().shape
    pad = (k_shape[-2]/2, k_shape[-1]/2)
    # support the kernel stride. This is necessary for AlexNet
    K = theano.tensor.nnet.conv2d(A, k, border_mode=pad)

    return bin_conv_out, K
    


def binarize_fc_weights(W):
    # symbolic binary weight
    Wb = T.cast(T.switch(T.ge(W, 0),1,-1), theano.config.floatX)
    # BinaryNet method
    #Wb = T.cast(T.switch(T.round(hard_sigmoid(W)),1,-1), theano.config.floatX)

    alpha = T.mean(T.abs_(W), axis=0)
    return Wb, alpha

def binarize_fc_input(fc_input):   ####这里怎么不变形状成为向量呢？？？？？？？？？？？？？？是因为有其他地方去处理吗

    bin_out = binary_tanh_unit(fc_input)
    
    if(fc_input.ndim == 4):  # prev layer is conv or pooling. hence compute the l1 norm using all maps
        beta = T.mean(T.abs_(fc_input), axis=[1, 2, 3])

    else: # feeding layer is FC layer
        beta = T.mean(T.abs_(fc_input), axis=1)

    return bin_out, beta



class Conv2DLayer(lasagne.layers.Conv2DLayer):
    """ Binary convolution layer which performs convolution using XNOR and popcount operations.
    This is followed by the scaling with input and weight scaling factors K and alpha respectively.
    """

    #def __init__(self, incoming, num_filters, filter_size, xnor=True, nonlinearity=lasagne.nonlinearities.identity, **kwargs):
    def __init__(self, incoming, num_filters, filter_size, xnor=True, **kwargs):

        """
        Parameters
        -----------
        incoming : layer or tuple
            Ipnut layer to this layer. If this is fed by a data layer then this is a tuple representing input dimensions.
        num_filters: int
            Number of 3D filters present in this layer = No of feature maps generated by this layer
        filter_size: tuple
            Filter size of this layer. Leading dimension is = no of input feature maps.
        """
        self.xnor = xnor    

        # average filter to compute scaling factor for activation
        no_inputs = incoming.output_shape[1]       #输入数据的channel
        shape = (num_filters, no_inputs, filter_size[0], filter_size[1])  #(num_of_filters,channel,height,width)


        num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])   #卷积核的元素个数
        num_units = int(np.prod(filter_size)*num_filters)  #？？？？？？？？？？
        self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units))) #？？？？？？？？

        if(self.xnor):
            super(Conv2DLayer, self).__init__(incoming,
                num_filters, filter_size, W=lasagne.init.Uniform((-1, 1)), **kwargs)  #xnor参数初始化保证在-1~1之间
            self.params[self.W] = set(['xnor'])   #？？？？？？？？
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)


        if self.xnor:
            # average filter to compute the activation scaling factor
            beta_filter = np.ones(shape=shape).astype(np.float32) / (no_inputs*filter_size[0]*filter_size[1])
            self.beta_filter = self.add_param(beta_filter, shape, name='beta_filter', trainable=False, regularizable=False)
            Wb = np.zeros(shape=self.W.shape.eval(), dtype=np.float32)
            #alpha = np.ones(shape=(num_filters,), dtype=np.float32)
            xalpha = lasagne.init.Constant(0.1)   #参数初始化方式
            
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

