# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:48:10 2020

@author: Yang Xu

Project #2: Articial Neural Networks
Usage: python3 COSC525_Project2.py binary_crossentropy 0.5 1 example1
"""

import sys
import numpy as np

loss_function = sys.argv[1]
learning_rate = float(sys.argv[2])
num_epoch=int(sys.argv[3])
input_data = sys.argv[4]


class Neuron:
    '''
    create neurons for fully connected layer
    neurons: neuron weights
    bias: bias weights
    Neuron stores all weights as well as gradients
    
    num_outputs: number of neurons
    num_inputs: dimenson of inputs
    if neuron_weights and bias_weights are not given, generate random weights
    and zero for bias
    '''
    def __init__(self,num_outputs=4,num_inputs = 2,neuron_weights=None,
                 bias_weights=None,neuron_gradient=None,bias_gradient=None):
        if neuron_weights is None:
            self.neurons = np.random.rand(num_inputs,num_outputs)
        else:
            self.neurons = neuron_weights
            
        if bias_weights is None:
            self.bias = np.zeros((1,num_outputs))
        else:
            self.bias = bias_weights
        
        if neuron_gradient is None:
            self.neuron_gradient = np.zeros((num_inputs,num_outputs))
        else:
            self.neuron_gradient = neuron_gradient
            
        if neuron_gradient is None:
            self.bias_gradient = np.zeros((1,num_outputs))
        else:
            self.bias_gradient = bias_gradient
        
    def calculate(self,learning_rate=None):
        ##update all weights at once
        self.neurons-=self.neuron_gradient*learning_rate
        self.bias-=self.bias_gradient*learning_rate
        
class Neuron2D:
    '''
    create 2d neurons for convolutional layer
    neurons: neuron weights
    bias: bias weights
    Neuron stores all weights as well as gradients
    if neuron_weights and bias_weights are not given, generate random weights
    and zero for bias
    '''
    def __init__(self,num_inputs=1,num_outputs = 3,neuron_size=(3,3),neuron_weights=None,
                 bias_weights=None,neuron_gradient=None,bias_gradient=None):
        if neuron_weights== None:
            ##Initialize kernel with normal distribution
            self.neurons = np.zeros((neuron_size[0],neuron_size[1],
                                     num_inputs,num_outputs))
            for i in range(num_outputs):
                stddev = 1/np.sqrt(np.prod(neuron_size))
                w = np.random.normal(loc = 0, scale = stddev, 
                                     size = neuron_size)
                for j in range(num_inputs):
                    self.neurons[:,:,j,i]=w
        else:
            self.neurons = neuron_weights
            
        if bias_weights == None:
            self.bias = np.zeros((num_outputs,1))
        else:
            self.bias = bias_weights
            
        if neuron_gradient is None:
            self.neuron_gradient = np.zeros((neuron_size[0],neuron_size[1],
                                             num_inputs,num_outputs))
        else:
            self.neuron_gradient = neuron_gradient
        
        if bias_gradient is None:
            self.bias_gradient = np.zeros((num_outputs,1))
        else:
            self.bias_gradient = bias_gradient
        
    def calculate(self,learning_rate=None):
        ##update all weights at once
        self.neurons-=self.neuron_gradient*learning_rate
        self.bias-=self.bias_gradient*learning_rate
            
class FullyConnectedLayer:
    '''
    logistic and linear are the only two activation function supported here.
    layer: consist of Neuron class that stores weights and bias
    layer_outputs: store outputs calculated by this layer
    layer_inputs: store inputs that come into this layer
    '''
    def __init__(self,num_inputs=1, num_outputs=4,activation = "logistic"):
        self.activation = activation
        self.layer = Neuron(num_inputs=num_inputs,num_outputs=num_outputs)
        self.layer_outputs=None
        self.layer_inputs=None
        
    def calculate(self,inputs):
        
        self.layer_inputs=inputs.copy()
        neuron_weights = self.layer.neurons
        bias_weights = self.layer.bias
        outputs = np.dot(inputs,neuron_weights)+bias_weights
        if self.activation == "linear":
            outputs = outputs
        elif self.activation == "logistic":
            ##sigmoid function
            outputs = 1/(1+np.exp(-outputs))
        self.layer_outputs=outputs
        return outputs
    
    def backpropagation(self,delta):
        ##back propation for fully connected layer
        ##delta: the cumulative delta arriving at this layer
        
        ##calculated d(out)/d(net)
        if self.activation == "logistic":
            delta_net = self.layer_outputs*(1-self.layer_outputs)
        else:
		#when activation function is linear
            delta_net = np.array([1]).reshape(1,1)
        
        ##sum weighted delta from next layer
        delta = delta * delta_net

        ##caculate gradient for this layer
        neuron_gradient = self.layer_inputs.T*delta
        bias_gradient = delta
        self.layer.neuron_gradient = np.transpose(neuron_gradient.copy())
        self.layer.bias_gradient = bias_gradient.copy()
        
        ##calculate cumulated delta and send it back to previous layer
        #sum_of_delta = np.sum(self.layer.neurons*delta)
        sum_of_delta = self.layer.neurons*delta
        return sum_of_delta

class ConvolutionalLayer:
    '''
    logistic and linear are the only two activation function supported here.
    layer: consist of Neuron2D class that stores weights and bias
    layer_outputs: store outputs calculated by this layer
    layer_inputs: store inputs that come into this layer
    '''
    def __init__(self,input_channel=1,output_channel=4,
                 filter_size=(3,3),activation = "logistic"):
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.filter_size = filter_size
        self.activation = activation
        self.layer = Neuron2D(num_inputs=input_channel,
                              num_outputs=output_channel,
                              neuron_size=filter_size)
        self.layer_outputs=None
        self.layer_inputs=None
        
    def calculate(self,inputs):
        self.layer_inputs=inputs.copy()
        x,y = self.filter_size
        n,m,c = inputs.shape
        outputs = np.zeros((n-x+1,m-y+1,self.output_channel))
        for k in range(self.output_channel):
            for i in range(x//2,n-x//2):
                for j in range(y//2,m-y//2):
                    patch = inputs[i-x//2:i+x//2+1,j-y//2:j+y//2+1,:]
                    output = np.sum(patch*self.layer.neurons[:,:,:,k])+self.layer.bias[k,0]
                    outputs[i-x//2,j-y//2,k]=output
        if self.activation == "linear":
            outputs = outputs
        elif self.activation == "logistic":
            outputs = 1/(1+np.exp(-outputs))
        self.layer_outputs=outputs
        return outputs
    
    def backpropagation(self,delta):
        ##back propation for convolutional layer
        ##delta: the cumulative delta arriving at this layer
        
        neurons = self.layer.neurons
        
        ##derivative of net
        if self.activation == "logistic":
            delta_net = self.layer_outputs*(1-self.layer_outputs)
        else:
		#when activation function is linear
            delta_net = np.array([1]).reshape(1,1)
        delta = delta * delta_net
        
        ##caculate gradient for this layer
        gradient=np.zeros((neurons.shape))
        x,y,o = delta.shape
        n,m,c = self.layer_inputs.shape
        for k in range(o):
            for i in range(x//2,n-x//2):
                for j in range(y//2,m-y//2):
                    for g in range(c):
                        if x%2==0 and y%2==0:
                            patch = self.layer_inputs[i-x//2:i+x//2,j-y//2:j+y//2,g]
                        else:
                            patch = self.layer_inputs[i-x//2:i+x//2+1,j-y//2:j+y//2+1,g]
                        output = np.sum(patch*delta[:,:,k])
                        gradient[i-x//2,j-y//2,g,k]=output
        self.layer.neuron_gradient = gradient.copy()
        self.layer.bias_gradient = np.sum(np.sum(delta,axis=0),axis=0)
        
        ##calculate cumulated delta and send it back to previous layer 
        neurons = np.flip(neurons)
        pad_delta = np.zeros((n+self.filter_size[0]-1,m+self.filter_size[1]-1,c))
        for i in range(o):
            ##pad delta to same size of input
            pad_delta[:,:,i]= np.pad(delta[:,:,i],(self.filter_size[0]-1,self.filter_size[1]-1),
                                     mode="constant")
        sum_of_delta=np.zeros((self.layer_inputs.shape))
        w,h,_,_ = self.layer.neurons.shape
        n,m,c = pad_delta.shape
        for k in range(c):
            for i in range(w//2,n-w//2):
                for j in range(h//2,m-h//2):
                    for g in range(o):
                        if w%2==0 and h%2==0:
                            patch = pad_delta[i-w//2:i+w//2,j-h//2:j+h//2,g]
                        else:
                            patch = pad_delta[i-w//2:i+w//2+1,j-h//2:j+h//2+1,g]
                        output = np.sum(patch*neurons[:,:,k,g])
                        sum_of_delta[i-w//2,j-h//2,k]+=output
        
        return sum_of_delta
    
class MaxPoolingLayer:
    '''
    2X2 Max pooling with stride 2 to downsample output size
    layer_outputs: store outputs calculated by this layer
    mask: mark the locations of max values
    '''
    def __init__(self):

        self.layer_outputs=None
        self.mask = None
        
    def calculate(self,inputs):
        n,m,c = inputs.shape
        h,w = int((n-2)/2+1),int((m-2)/2+1)
        outputs = np.zeros((h,w,c))
        mask = np.zeros((n,m,c))
        for k in range(c):
            for i in range(h):
                for j in range(w):
                    patch = inputs[i*2:i*2+2,j*2:j*2+2,k]
                    outputs[i,j,k]=np.max(patch)
                    mask[i*2+np.where(patch == np.max(patch))[0][0],
                         j*2+np.where(patch == np.max(patch))[1][0],k]=1
                    
        self.layer_outputs=outputs
        self.mask = mask
        return outputs
    
    def backpropagation(self,delta):
        ##back propation for maxpooling layer
        ##delta: the cumulative delta arriving at this layer
        w,h,c = delta.shape
        gradient = np.zeros((w*2,h*2,c))
        for k in range(c):
            for i in range(w):
                for j in range(h):
                    patch = self.mask[i*2:i*2+2,j*2:j*2+2,k]
                    gradient[i*2+np.where(patch == np.max(patch))[0][0],
                             j*2+np.where(patch == np.max(patch))[1][0]]=delta[i,j,k]
        return gradient

class FlattenLayer:
    '''
    Flatten output from convolutional layer to 1D vector
    layer_outputs: store outputs calculated by this layer
    '''
    def __init__(self):
        self.layer_outputs=None
        self.channel = None
        self.dim1 = None
        self.dim2 = None
    def calculate(self,inputs):
        self.dim1,self.dim2, self.channel = inputs.shape
        self.layer_outputs= inputs.flatten().copy()
        return inputs.flatten()
    def backpropagation(self,delta):
        ##back propation for flatten layer
        ##reshape delta to 3D array
        new_delta = delta.reshape(self.dim1,self.dim2,self.channel)
        return new_delta
        
class NeuralNetwork:
    '''
    lossfunc: loss function (MSE or binary_crossentropy)
    activation: activation function (linear or logistic)
    learning_rate: model learning rate
    NeuralNetwork contain net that is an empty list when it is just created
    addLayer function can add specific layer to the list of net
    '''
    def __init__(self,loss_function="binary_crossentropy",learning_rate=0.01):
     
       self.lossfunc = loss_function
       self.learning_rate= learning_rate
       self.net=[]
       self.layer_name=[]
       
    def addLayer(self,layer_names=None,inputs_dim=None,outputs_dim=None,
                 filter_size=None,activation_func=None):
        if layer_names=="FullyConnectedLayer":
            self.net.append(FullyConnectedLayer(num_inputs=inputs_dim, 
                                                num_outputs=outputs_dim,
                                                activation = activation_func))
            self.layer_name.append(layer_names)
        elif layer_names=="ConvolutionalLayer":
            self.net.append(ConvolutionalLayer(input_channel=inputs_dim,
                                               output_channel=outputs_dim,
                                               filter_size=filter_size,
                                               activation = activation_func))
            self.layer_name.append(layer_names)
        elif layer_names=="MaxPoolingLayer":
            self.net.append(MaxPoolingLayer())
            self.layer_name.append(layer_names)
        elif layer_names=="FlattenLayer":
            self.net.append(FlattenLayer())
            self.layer_name.append(layer_names)
            
    def calculate(self,inputs):
        ##forward model
        ##loop every layer to calculate output
        for layer in self.net:
            outputs=layer.calculate(inputs)
            inputs = outputs.copy()
        return outputs
    
    def calculateloss(self,outputs, targets):
        if self.lossfunc =="MSE":
            return (np.square(targets - outputs)).mean(axis=0)
        
        elif self.lossfunc =="binary_crossentropy":
            return (-targets*np.log(outputs)-(1-targets)*np.log(1-outputs)).mean(axis=0)
    
    def backward(self,targets,outputs):##backward model
        #loop layers from the output layer to the input layer
        delta=1
        if self.lossfunc =="binary_crossentropy":
            delta = delta * (-targets/outputs+(1-targets)/(1-outputs))    
        else:
            delta = delta * np.mean(2*(outputs-targets))
        
        for i in range(len(self.net)-1,-1,-1):
            delta=self.net[i].backpropagation(delta)##calculate delta and send it back to previous layer
            
    def train(self,data=None,targets=None,num_epoch=1000):
        losses =[]
        ## dimension of data
        s,n,m,c = data.shape
        ## dimension of targets
        S,M = targets.shape
        for i in range(num_epoch):
            loss=0
            ##iterate each data point
            for j in range(s):
                x = data[j,:].reshape(n,m,c)
                y= targets[j,:].reshape(1,M)
                outputs=self.calculate(x)
                loss+=self.calculateloss(outputs,y)
                self.backward(y,outputs)
                ##update weights in each layer
                for k in range(len(self.net)-1,-1,-1):
                    if self.layer_name[k] not in ["MaxPoolingLayer","FlattenLayer"]:
                        self.net[k].layer.calculate(learning_rate=self.learning_rate)        
            losses.append(np.mean(loss/data.shape[0]))
        return losses
        
def main():
    
    if input_data=="example1":
        data=np.array([[1,2,3,4,5],[5,4,3,2,1],[1,2,3,4,5],[5,4,3,2,1],[1,2,3,4,5]])
        data=data.reshape(1,5,5,1)
        y=np.zeros((1,1))
        convnet=NeuralNetwork()
        layers = {"layer1":["ConvolutionalLayer",1,1,(3,3),"logistic"],
                  "layer2":["FlattenLayer",None,None,None,None],
                  "layer3":["FullyConnectedLayer",9,1,None,"logistic"]}
        for l,parameters in layers.items():
            p1,p2,p3,p4,p5 = parameters
            convnet.addLayer(layer_names=p1,inputs_dim=p2,outputs_dim=p3,
                         filter_size=p4,activation_func=p5)
    elif input_data=="example2":
        data=np.array([[1,2,3,4,5],[5,4,3,2,1],[1,2,3,4,5],[5,4,3,2,1],[1,2,3,4,5]])
        data=data.reshape(5,5,1)
        y=np.zeros((1,1))
        convnet=NeuralNetwork()
        layers = {"layer1":["ConvolutionalLayer",1,1,(3,3),"logistic"],
                  "layer2":["ConvolutionalLayer",1,1,(3,3),"logistic"],
                  "layer3":["FlattenLayer",None,None,None,None],
                  "layer4":["FullyConnectedLayer",1,1,None,"logistic"]}
        for l,parameters in layers.items():
            p1,p2,p3,p4,p5 = parameters
            convnet.addLayer(layer_names=p1,inputs_dim=p2,outputs_dim=p3,
                         filter_size=p4,activation_func=p5)
    elif input_data =="example3":
        data=np.array([[1,2,3,4,5,6],[6,5,4,3,2,1],[1,2,3,4,5,6],
                       [6,5,4,3,2,1],[1,2,3,4,5,6],[6,5,4,3,2,1]])
        data=data.reshape(6,6,1)
        y=np.zeros((1,1))
        convnet=NeuralNetwork()
        layers = {"layer1":["ConvolutionalLayer",1,1,(3,3),"logistic"],
                  "layer2":["MaxPoolingLayer",None,None,None,None],
                  "layer3":["FlattenLayer",None,None,None,None],
                  "layer4":["FullyConnectedLayer",4,1,None,"logistic"]}
        for l,parameters in layers.items():
            p1,p2,p3,p4,p5 = parameters
            convnet.addLayer(layer_names=p1,inputs_dim=p2,outputs_dim=p3,
                         filter_size=p4,activation_func=p5)

    losses = convnet.train(data=data,targets=y,num_epoch=num_epoch)
    losses = np.array(losses)
    np.savetxt("training_loss.txt",losses,fmt='%.5f')

if __name__ == '__main__':
    main()