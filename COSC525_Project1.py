# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:48:10 2020

@author: Yang Xu

Project #1: Articial Neural Networks
Usage: python3 COSC525_Project1.py binary_crossentropy logistic 0.1 2,4,1 10000 xor
"""

import sys
import numpy as np
import pandas as pd

loss_function = sys.argv[1]
activation_function = sys.argv[2]
learning_rate = float(sys.argv[3])
layers = list(sys.argv[4].split(','))
layers = [int(i) for i in layers]
num_epoch=int(sys.argv[5])
training_set = sys.argv[6]


class Neuron:
    '''
    create neurons in neural networks
    neurons: neuron weights
    bias: bias weights
    Neuron stores all weights as well as gradients
    
    num_outputs: number of neurons
    num_inputs: dimenson of inputs
    if neuron_weights and bias_weights are not given, generate random weights
    '''
    def __init__(self,num_outputs=4,num_inputs = 2,neuron_weights=None,
                 bias_weights=None,gradient=None):
        if neuron_weights is None:
            self.neurons = np.random.rand(num_inputs,num_outputs)
        else:
            self.neurons = neuron_weights
            
        if bias_weights is None:
            self.bias = np.random.rand(1,num_outputs)
        else:
            self.bias = bias_weights
        
        if gradient is None:
            self.gradient = np.zeros((num_inputs+1,num_outputs))
        else:
            self.gradient = gradient
        
    def calculate(self):
        ##update all weights at once
        self.neurons-=self.gradient[:-1,:]
        self.bias-=self.gradient[-1,:]

            
class FullyConnectedLayer:
    '''
    logistic and linear are the only two activation function supported here.
    num_layers: number of layers that includes input and output layer. 
    num_nodes: a list that defines how many neurons in each layer;
    the first item in this list indicates the dimension of inputs,
    and the last item is the number of targets.
    '''
    def __init__(self,num_layers=3, num_nodes=[2,4,1],activation = "logistic"):
        self.activation = activation
        self.layers = [Neuron(num_inputs=num_nodes[i],num_outputs=num_nodes[i+1]) for i in range(num_layers-1)]
        self.layer_outputs=None
        
    def calculate(self,inputs):
        ##create a list to store outputs of net
        self.layer_outputs = []
        ##suppose the data out from a hidden layer that is not in the net
        self.layer_outputs.append(inputs)
        
        for l in self.layers:
            neuron_weights = l.neurons
            bias_weights = l.bias
            outputs = np.dot(inputs,neuron_weights)+bias_weights
            if self.activation == "linear":
                outputs = outputs
            elif self.activation == "logistic":
                ##sigmoid function
                outputs = 1/(1+np.exp(-outputs))
            self.layer_outputs.append(outputs)
            inputs = outputs.copy()
        return outputs

        
class NeuralNetwork:
    '''
    lossfunc: loss function (MSE or binary_crossentropy)
    activation: activation function (linear or logistic)
    learning_rate: model learning rate
    num_nodes_of_layer: a list that defines how many nodes of each layer
    '''
    def __init__(self,num_nodes_of_layer=[2,4,1],activation = "logistic",
                 loss_function="binary_crossentropy",learning_rate=0.01):
       self.net = FullyConnectedLayer(num_layers=len(num_nodes_of_layer),
                                      num_nodes=num_nodes_of_layer)
       self.lossfunc = loss_function
       self.activation = activation
       self.learning_rate= learning_rate
       
    def calculate(self,inputs):##forward model
        return self.net.calculate(inputs)
    
    def calculateloss(self,outputs, targets):
        if self.lossfunc =="MSE":
            return (np.square(targets - outputs)).mean(axis=0)
        
        elif self.lossfunc =="binary_crossentropy":
            return (-targets*np.log(outputs)-(1-targets)*np.log(1-outputs)).mean(axis=0)
    
    def backpropation(self,targets):##backward model
        #loop layers from the output layer to the input layer
        delta=1
        for i in range(len(self.net.layers)-1,-1,-1):
            out_layer=self.net.layer_outputs[i+1]##outputs of this layer
            in_layer = self.net.layer_outputs[i]##inputs of this layer 
            
            ##calculated d(out)/d(net)
            if self.activation == "logistic":
                delta_net = out_layer*(1-out_layer)
            else:
		#when activation function is linear
                delta_net = np.array([1]).reshape(1,1)
                
            if i<len(self.net.layers)-1:
                ##calculate delta for any layers before the last one
                sum_of_delta = np.sum(self.net.layers[i+1].neurons*delta)##sum weighted delta from next layer
                delta = sum_of_delta * delta_net
            else:
                ##calculate delta for last layer
                delta = delta * (-targets/out_layer+(1-targets)/(1-out_layer))*delta_net
            ##the gradient is multiplied by learning rate
            gradient = np.concatenate((delta*in_layer.T,delta),axis=0)
            self.net.layers[i].gradient = self.learning_rate*gradient
            
    def train(self,data=None,targets=None,num_epoch=1000):
        losses =[]
        ## dimension of data
        n,m = data.shape
        ## dimension of targets
        N,M = targets.shape
        for i in range(num_epoch):
            loss=0
            ##iterate each data point
            for j in range(n):
                x = data[j,:].reshape(1,m)
                y= targets[j,:].reshape(1,M)
                outputs=self.calculate(x)
                loss+=self.calculateloss(outputs,y)
                self.backpropation(y)
                ##update weights in each layer
                for k in range(len(self.net.layers)-1,-1,-1):
                    self.net.layers[k].calculate()        
            losses.append(np.mean(loss/data.shape[0]))
        return losses
        
    
def main():
    
    if training_set=="and":
        data = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([0,0,0,1]).reshape(4,1)
    elif training_set=="xor":
        data = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([0,1,1,0]).reshape(4,1)
    else:
        ##data should be tab-saparated txt file with no header and index column
        ##the last column should target value
        data = pd.read_csv(sys.argv[7],header=None,index_col=False,sep="\t")
        y = data.iloc[:,-1].values
        data = data.iloc[:,:-1].values
        
    #model = NeuralNetwork(num_nodes_of_layer=[2,4,1],
    #                      activation = "logistic",
    #                      loss_function="binary_crossentropy",
    #                      learning_rate=1)
    model = NeuralNetwork(num_nodes_of_layer=layers,
                          activation = activation_function,
                          loss_function=loss_function,
                          learning_rate=learning_rate)
    losses = model.train(data=data,targets=y,num_epoch=num_epoch)
    losses = np.array(losses)
    np.savetxt("training_loss.txt",losses,fmt='%.5f')

if __name__ == '__main__':
    main()