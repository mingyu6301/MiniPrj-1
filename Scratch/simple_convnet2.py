# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class SimpleConvNet2:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param1={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param2={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param3={'filter_num': 128, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=256, output_size=10, weight_init_std='he', dropout_ratio=0.5):

        # 출력 크기 계산 
        filter_num1, filter_size1 = conv_param1['filter_num'], conv_param1['filter_size']
        filter_pad1, filter_stride1 = conv_param1['pad'], conv_param1['stride']
        conv_out1 = (input_dim[1] - filter_size1 + 2*filter_pad1) / filter_stride1 + 1
        pool_out1 = int(conv_out1 / 2)

        filter_num2, filter_size2 = conv_param2['filter_num'], conv_param2['filter_size']
        filter_pad2, filter_stride2 = conv_param2['pad'], conv_param2['stride']
        conv_out2 = (pool_out1 - filter_size2 + 2*filter_pad2) / filter_stride2 + 1
        pool_out2 = int(conv_out2 / 2)

        filter_num3, filter_size3 = conv_param3['filter_num'], conv_param3['filter_size']
        filter_pad3, filter_stride3 = conv_param3['pad'], conv_param3['stride']
        conv_out3 = (pool_out2 - filter_size3 + 2*filter_pad3) / filter_stride3 + 1
        fc_input_size = int(filter_num3 * conv_out3 * conv_out3)

        # He 초깃값
        def get_scale(init_std, fan_in):
            if str(init_std).lower() in ('relu', 'he'):
                return np.sqrt(2.0 / fan_in)
            elif str(init_std).lower() in ('sigmoid', 'xavier'):
                return np.sqrt(1.0 / fan_in)
            else:
                return init_std

        #계산
        fan_in1 = input_dim[0] * filter_size1 * filter_size1
        fan_in2 = filter_num1 * filter_size2 * filter_size2
        fan_in3 = filter_num2 * filter_size3 * filter_size3
        fan_in4 = fc_input_size
        fan_in5 = hidden_size

        # 가중치 초기화 ('he'를 동적으로 적용)
        self.params = {}
        self.params['W1'] = get_scale(weight_init_std, fan_in1) * np.random.randn(filter_num1, input_dim[0], filter_size1, filter_size1)
        self.params['b1'] = np.zeros(filter_num1)
        self.params['W2'] = get_scale(weight_init_std, fan_in2) * np.random.randn(filter_num2, filter_num1, filter_size2, filter_size2)
        self.params['b2'] = np.zeros(filter_num2)
        self.params['W3'] = get_scale(weight_init_std, fan_in3) * np.random.randn(filter_num3, filter_num2, filter_size3, filter_size3)
        self.params['b3'] = np.zeros(filter_num3)
        self.params['W4'] = get_scale(weight_init_std, fan_in4) * np.random.randn(fc_input_size, hidden_size)
        self.params['b4'] = np.zeros(hidden_size)
        self.params['W5'] = get_scale(weight_init_std, fan_in5) * np.random.randn(hidden_size, output_size)
        self.params['b5'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1']   = Convolution(self.params['W1'], self.params['b1'], conv_param1['stride'], conv_param1['pad'])
        self.layers['Relu1']   = Relu()
        self.layers['Pool1']   = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Conv2']   = Convolution(self.params['W2'], self.params['b2'], conv_param2['stride'], conv_param2['pad'])
        self.layers['Relu2']   = Relu()
        self.layers['Pool2']   = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Conv3']   = Convolution(self.params['W3'], self.params['b3'], conv_param3['stride'], conv_param3['pad'])
        self.layers['Relu3']   = Relu()
        self.layers['Affine1'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Relu4']   = Relu()
        self.layers['Dropout1'] = Dropout(dropout_ratio)
        self.layers['Affine2'] = Affine(self.params['W5'], self.params['b5'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for name, layer in self.layers.items():
            if name == 'Dropout1':
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W4'], grads['b4'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W5'], grads['b5'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads
