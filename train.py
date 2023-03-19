#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import notebook 
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs
import wandb
from sklearn.metrics import mean_squared_error


# In[1]:


#get_ipython().system('pip install wandb')


# In[2]:


class MultiClass_classification:
  
  def __init__(self,input_layer, hidden_sizes , output_layer):
    
    
    self.sizes = [input_layer] + hidden_sizes + [output_layer] 

    self.W = {}
    self.B = {}
    self.no_hidden = len(hidden_sizes)
    for i in range(self.no_hidden+1):
      #self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
      #self.B[i+1] = np.zeros((1, self.sizes[i+1]))
      self.W[i+1] =  np.random.uniform(-1,1,(self.sizes[i], self.sizes[i+1]))
      #self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
      self.B[i+1] = np.random.normal(-1,1,(1,self.sizes[i+1]))  
      #self.B[i+1] = np.zeros((1, self.sizes[i+1]))  
      
  def sigmoid(self, x):
    return 1.0/(1.0 + np.exp(-x))

  def relu(self, x):
   return np.maximum(0, x)

  def tanh(self, x):
    return np.tanh(x)

  def grad_sigmoid(self, x):
    return x*(1-x) 

  def grad_relu(self, x):
    return 1.0*(x>0)
    
  def grad_tanh(self, x):
    return 1 - np.power(np.tanh(x),2)

  def softmax(self, x):
    #print(x)
    exps = np.exp(x)
    return exps /  np.maximum ( 0.01 , np.sum(exps) )
  def set_activation(self):
    mapping = {}
    self.A = {}
    mapping["D_A"] = []
    mapping["D_H"] = []
    self.H = {}   
  def forward_pass(self, x):
    self.set_activation()
    self.H[0] = x.reshape(1, -1)
    for i in range(self.no_hidden):
      temp_H =  self.H[i] 
      temp_W =  self.W[i+1] 
      temp_B =  self.B[i+1] 
      self.A[i+1] = np.matmul(temp_H, temp_W) + temp_B
      if self.activation_function == "sigmoid":
          self.H[i+1] = self.sigmoid(self.A[i+1])
      if self.activation_function == "relu":  
          #print("relu_2") 
          self.H[i+1] = self.relu(self.A[i+1])
      if self.activation_function == "tanh":
          #print("tanh")
          self.H[i+1] = self.tanh(self.A[i+1])
    temp_H = self.H[self.no_hidden] 
    temp_W =self.W[self.no_hidden+1]
    temp_B = self.B[self.no_hidden+1]
    self.A[self.no_hidden+1] = np.matmul(temp_H, temp_W) + temp_B
    self.H[self.no_hidden+1] = self.softmax(self.A[self.no_hidden+1] - np.max(self.A[self.no_hidden+1]))
    #x - np.max(x)
    return self.H[self.no_hidden+1]
  
  def predict(self, X):
    X_data = X
    z =  X_data.shape[1]
    Y_pred = []
    for x in X:
      Y_pred.append(self.forward_pass(x))
    predict = Y_pred
    return np.array(predict).squeeze()
 
  """    Question 8     """
  def square_error_loss(self,label,pred):
    #print("mse")
    temp = np.power((label - pred),2)
    temp1 = np.sum(temp)
    mse=np.mean(temp1)
    return mse
  def cross_entropy(self,label,pred):
    yl=np.multiply(pred,label)
    yl=yl[yl!=0]
    yl=-np.log(yl)
    yl=np.mean(yl)
    return yl
  def derivates(self):
    mapping= {}
    self.dW = {}
    mapping["D_W"] = []
    self.dB = {}
    mapping["D_B"] = []
    self.dH = {}
    mapping["D_H"] = []
    self.dA = {}
    mapping["D_A"] = []
    
  """    Question 3     """  
    
  def grad(self, x, y):
    self.forward_pass(x) 
    self.derivates()
    if(self.type_of_loss == "cross_entropy"):
        self.dA[self.no_hidden + 1] = (self.H[self.no_hidden + 1] - y)
    else:
       #delA.append(np.array((y_hat - ey)(y_hat - y_hat*2)))
       #print("yes")
       self.dA[self.no_hidden + 1] = (self.H[self.no_hidden + 1] - y)*(self.H[self.no_hidden + 1] - self.H[self.no_hidden + 1]**2)
    Layer = self.no_hidden + 1
    for k in range(Layer, 0, -1):
      self.dB[k] = self.dA[k]
      temp = self.H[k-1].T
      self.dW[k] = np.matmul(temp, self.dA[k])
      temp = self.W[k].T
      self.dH[k-1] = np.matmul(self.dA[k], temp)
      if self.activation_function == "sigmoid":
        temp = self.grad_sigmoid(self.H[k-1])
      if self.activation_function == "relu":
        #print("relu_1")
        temp = self.grad_relu(self.H[k-1])
      if self.activation_function == "tanh":
        #print("tanh")
        temp = self.grad_tanh(self.H[k-1])  
      self.dA[k-1] = np.multiply(self.dH[k-1], temp ) 
    
  def fit(self, X, Y, epochs=100, initialize='True', learning_rate=0.01, display_loss=False,batch_size = 512 , batch = 'b',algorithm = "gd",gamma = 0.09,act_func = "sigmoid",intialize_method = "random",loss_type = "cross_entropy",l2_norm = True,lambda_value = 500):
    Layer = self.no_hidden+1  
    self.activation_function = act_func
    self.type_of_loss = loss_type
    #for i in range(0,Layer,1):
     #   self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
     #   self.B[i+1] = np.zeros((1, self.sizes[i+1]))
    if(intialize_method == "random"):
        for i in range(Layer):

          self.W[i+1] =  np.random.uniform(-1,1,(self.sizes[i], self.sizes[i+1]))
          self.B[i+1] = np.random.normal(-1,1,(1,self.sizes[i+1]))
    he = 1
    if(intialize_method == "xavier"):
        if(act_func == "relu"):
            he = 2
        for i in range(Layer):

          self.W[i+1] =  np.random.uniform(-1,1,(self.sizes[i], self.sizes[i+1]))*np.sqrt(he/self.sizes[i-1])
     
          self.B[i+1] = np.random.normal(-1,1,(1,self.sizes[i+1]))
      
    loss = {}  
    for epoch in tqdm(range(epochs), total=epochs, unit="epoch"):
      print('epoch',epoch)
      dW = {}
      number_of_layer = self.no_hidden+1
      for i in range(0,number_of_layer,1):
        dW[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))
      self.derviative_of_w = []
 
      dB = {}
      for i in range(0,number_of_layer,1):
        dB[i+1] = np.zeros((1, self.sizes[i+1]))
      self.derviative_of_b = []   
      v_w = {}
      v_b = {}
      self.v_w_hat = {}
      self.v_b_hat = {}
      self.m_w = {}
      self.m_b = {}
      self.m_w_hat = {}
      self.m_b_hat = {}
      for i in range(self.no_hidden+1):
        v_w[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))
        v_b[i+1] = np.zeros((1, self.sizes[i+1]))
        self.m_w[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))
        self.m_b[i+1] = np.zeros((1, self.sizes[i+1]))
        self.v_w_hat[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))
        self.v_b_hat[i+1] = np.zeros((1, self.sizes[i+1]))
        self.m_w_hat[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))
        self.m_b_hat[i+1] = np.zeros((1, self.sizes[i+1]))
      if algorithm == "NAG":
        print("did early work")
        for i in range(self.no_hidden+1):
            self.W[i+1] -= gamma * v_w[i+1]
            self.B[i+1] -= gamma * v_b[i+1]
      if(batch == 'b'):
          #print("b")
          learn_number = learning_rate   
          for x, y in zip(X, Y):
            Layer = self.no_hidden+1
            self.grad(x, y)
            for i in range(0,Layer,1):
              dW[i+1] = dW[i+1] +self.dW[i+1]     
              dB[i+1] = dB[i+1] +self.dB[i+1] 

          m = X.shape[1]
          v_w,v_b = self.optimizer(dW,dB,m,v_w,v_b,algo = algorithm,learning_rate = learn_number)
          """
          for i in range(self.no_hidden+1):
            temp = learning_rate * (dW[i+1]/m)
            self.W[i+1] =self.W[i+1] -  temp
            temp = learning_rate * (dB[i+1]/m)
            self.B[i+1] =self.B[i+1] - temp
          """  
            
        
      if(batch == 's'):
          m = X.shape[1]
          learn_number = learning_rate   
          for x, y in zip(X, Y):
            Layer = self.no_hidden+1
            self.grad(x, y)
            for i in range(0,Layer,1):
              
              dW[i+1] = self.dW[i+1]     
              dB[i+1] = self.dB[i+1]
              v_w,v_b = self.optimizer(dW,dB,m,v_w,v_b,algo = algorithm,learning_rate = learn_number)
              """
              temp = learning_rate * (dW[i+1]/m)
              self.W[i+1] =self.W[i+1] -  temp
              temp = learning_rate * (dB[i+1]/m)
              self.B[i+1] =self.B[i+1] - temp
              """
                
      
      if(batch == "m_B"):
          var = 0
          learn_number = learning_rate 
          for x, y in zip(X, Y):
            var+=1
            self.grad(x, y)
            m = X.shape[1]
            data_size = m
            for i in range(0,self.no_hidden+1,1):
                
              dW[i+1] =dW[i+1] + self.dW[i+1]
              dB[i+1] =dB[i+1] + self.dB[i+1]
              if(var%batch_size == 0)  :
                  v_w,v_b = self.optimizer(dW,dB,m,v_w,v_b,algo = algorithm,learning_rate = learn_number)
                  #self.W[i+1] -= learning_rate * (dW[i+1]/m)
                  #self.B[i+1] -= learning_rate * (dB[i+1]/m)
                  dW[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))
                  dB[i+1] = np.zeros((1, self.sizes[i+1])) 
                    
      m = X.shape[1]
      flag = 0
      if l2_norm == True:
          if flag == 0:
                print('Yes')
                flag = 1
          for i in range(self.no_hidden+1):
             self.W[i+1] -=  ((learning_rate * (lambda_value)/m)* self.W[i+1])
        
        
      if loss_type == "cross_entropy"  :    
          loss[epoch] = self.cross_entropy(Y, self.predict(X))
      if loss_type == "square_error_loss"  :    
          loss[epoch] = self.square_error_loss(Y, self.predict(X))  
      
             
      wandb.log({"loss": loss[epoch] , "epoch": epochs})
    self.show_loss(loss)
    # return 
    
    
  """    Question 3     """


  def optimizer(self,dW,dB,m,v_w,v_b,algo = "gd",gamma = 0.9,learning_rate = 0.001,beta = 0.95,epsilon = 0.000000001):
    #print("called")
    if(algo == "gd"):
        
        for i in range(self.no_hidden+1):
           self.W[i+1] = self.W[i+1] - learning_rate * (dW[i+1]/m)
           self.B[i+1] = self.B[i+1] - learning_rate * (dB[i+1]/m)
        return v_w,v_b

    if(algo == "momentum"):
     
        for i in range(self.no_hidden+1):
             v_w[i+1] = gamma *v_w[i+1] + learning_rate * (dW[i+1]/m)
             v_b[i+1] = gamma *v_b[i+1] + learning_rate * (dB[i+1]/m)
             self.W[i+1] -= v_w[i+1]
             self.B[i+1] -= v_b[i+1]
        return v_w,v_b 
    if(algo == "NAG"):
        for i in range(self.no_hidden+1):
            self.W[i+1] -= learning_rate * (dW[i+1]/m)
            self.B[i+1] -= learning_rate * (dB[i+1]/m)
            v_w[i+1] = gamma * v_w[i+1] + learning_rate * (dW[i+1]/m)
            v_b[i+1] = gamma * v_b[i+1] + learning_rate * (dB[i+1]/m)
        return v_w,v_b
    if(algo == "RMSProp"):
        lr = learning_rate * 5
        for i in range(self.no_hidden+1):
            v_w[i+1] = (beta * v_w[i+1]) + ((1 - beta) * ((dW[i+1]/m) ** 2 ) )
            v_b[i+1] = (beta * v_b[i+1]) + ((1 - beta) * ((dB[i+1]/m) ** 2 ))
            self.W[i+1] -= (((lr)/((v_w[i+1] ** 0.5) + epsilon)) * (dW[i+1]/m))
            self.B[i+1] -= (((lr)/((v_b[i+1] ** 0.5) + epsilon)) * (dB[i+1]/m))
        return v_w,v_b
    if(algo == "Adam"):
        beta_1=0.9
        beta_2=0.9
        time = 0
        lr = 0.8
        for i in range(self.no_hidden+1):
            time += 1
            self.m_w[i+1] = beta_1 * self.m_w[i+1] + (1 - beta_1) * (dW[i+1]/m)
            self.m_b[i+1] = beta_1 * self.m_b[i+1] + (1 - beta_1) * (dB[i+1]/m)
            v_w[i+1] = beta_2 * v_w[i+1] + (1 - beta_2) * ((dW[i+1]/m) ** 2 ) 
            v_b[i+1] = beta_2 * v_b[i+1] + (1 - beta_2) * ((dB[i+1]/m) ** 2 )
            self.m_w_hat[i+1] = self.m_w[i+1] / (1 - np.power(beta_1, time))
            self.m_b_hat[i+1] = self.m_b[i+1] / (1 - np.power(beta_1, time))
            self.v_w_hat[i+1] = v_w[i+1] / (1 - np.power(beta_2, time))
            self.v_b_hat[i+1] = v_b[i+1] / (1 - np.power(beta_2, time))
            self.W[i+1] -= ((lr)/((self.v_w_hat[i+1] ** 0.5) + epsilon) * (self.m_w_hat[i+1]/m))
            self.B[i+1] -= ((lr)/((self.v_b_hat[i+1] ** 0.5) + epsilon ) * (self.m_b_hat[i+1]/m))
            
            
        return v_w,v_b
    if(algo == "Nadam"):
        beta_1=0.9
        beta_2=0.9
        time = 0
        lr = 0.0001
        for i in range(self.no_hidden+1):
            
            self.m_w[i+1] = beta_1 * self.m_w[i+1] + (1 - beta_1) * (dW[i+1]/m)
            self.m_b[i+1] = beta_1 * self.m_b[i+1] + (1 - beta_1) * (dB[i+1]/m)
            v_w[i+1] = beta_2 * v_w[i+1] + (1 - beta_2) * ((dW[i+1]/m) ** 2 ) 
            v_b[i+1] = beta_2 * v_b[i+1] + (1 - beta_2) * ((dB[i+1]/m) ** 2 )
            self.m_w_hat[i+1] = self.m_w[i+1] / (1 - np.power(beta_1, time+1))
            self.m_b_hat[i+1] = self.m_b[i+1] / (1 - np.power(beta_1, time+1))
            self.v_w_hat[i+1] = v_w[i+1] / (1 - np.power(beta_2, time+1))
            self.v_b_hat[i+1] = v_b[i+1] / (1 - np.power(beta_2, time+1))
            self.W[i+1] -= ((lr/np.sqrt(self.v_w_hat[i+1] + epsilon)) * (beta_1 * self.m_w_hat[i+1] + (1- beta_1)*(dW[i+1]/m)/(1 - np.power(beta_1 ,time+1))))
            self.B[i+1] -= ((lr/np.sqrt(self.v_b_hat[i+1] + epsilon)) * (beta_1 * self.m_b_hat[i+1] + (1- beta_1)*(dB[i+1]/m)/(1 - np.power(beta_1 ,time+1))))
            time += 1
            
        return v_w,v_b   
  def show_loss(self,loss):
      plt.plot(np.array(list(loss.values())).astype(float))
      plt.xlabel('Epochs')
      plt.ylabel('CE')
      plt.show()


# In[3]:
parser = argparse.ArgumentParser()
parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='myprojectname')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='myname')
parser.add_argument('-d', '--dataset', help='choices: ["mnist", "fashion_mnist"]', type=str, default='fashion_mnist')
parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=5)
parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=32)
parser.add_argument('-l','--loss', help = 'choices: ["square_error_loss", "cross_entropy"]' , type=str, default='cross_entropy')
parser.add_argument('-o', '--optimizer', help = 'choices: ["gd", "momentum", "NAG", "RMSProp", "Adam", "Nadam"]', type=str, default = 'Nadam')
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.005)
parser.add_argument('-m', '--momentum', help='Momentum used by momentum and nag optimizers.',type=float, default=0.5)
parser.add_argument('-beta', '--beta', help='Beta used by rmsprop optimizer',type=float, default=0.5)
parser.add_argument('-beta1', '--beta1', help='Beta1 used by adam and nadam optimizers.',type=float, default=0.5)
parser.add_argument('-beta2', '--beta2', help='Beta2 used by adam and nadam optimizers.',type=float, default=0.5)
parser.add_argument('-eps', '--epsilon', help='Epsilon used by optimizers.',type=float, default=0.000001)
parser.add_argument('-w_d', '--weight_decay', help='Weight decay used by optimizers.',type=float, default=.0)
parser.add_argument('-w_i', '--weight_init', help = 'choices: ["random", "xavier"]', type=str, default='random')
parser.add_argument('-nhl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=3)
parser.add_argument('-sz', '--hidden_size', help ='Number of hidden neurons in a feedforward layer.', nargs='+', type=int, default=32, required=False)
parser.add_argument('-a', '--activation', help='choices: ["sigmoid", "tanh", "relu"]', type=str, default='tanh')
parser.add_argument('--hlayer_size', type=int, default=32)
parser.add_argument('-oa', '--output_activation', help = 'choices: ["softmax"]', type=str, default='softmax')
parser.add_argument('-oc', '--output_size', help ='Number of neurons in output layer used in feedforward neural network.', type = int, default = 10)
arguments = parser.parse_args()
if arguments.dataset == fashion_mnist:
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
else: 
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()   
train_data = np.array(X_train)
X_train = train_data / 255.0
test_data = np.array(X_test)
X_test = test_data / 255.0

print(train_data.shape)
print(test_data.shape)

encoder = OneHotEncoder()

X_train_len = int(len(X_train) * 0.9)
X1_train = X_train[0:X_train_len]
X2_train_validation = X_train[X_train_len:len(X_train)]
Y_train_len = int(len(Y_train) * 0.9)
Y1_train = Y_train[0:Y_train_len]
Y2_train_validation = Y_train[Y_train_len:len(Y_train)]

y_train = encoder.fit_transform(np.expand_dims(Y1_train,1)).toarray()
y_train_validation = encoder.fit_transform(np.expand_dims(Y2_train_validation,1)).toarray()
y_test = encoder.fit_transform(np.expand_dims(Y_test,1)).toarray()


wandb.login()
wandb.init(project=arguments.wandb_project) 
#wandb.init(project="assignment1_CS910") 
hidden_layer = []
for i in range(arguments.num_layers):
    hidden_layer.append(arguments.hidden_size)
print(hidden_layer)
object = MultiClass_classification(784, hidden_layer, 10)
enc = OneHotEncoder()
#ffsnn.fit(X1_train, y_train, epochs=5, learning_rate=0.005, batch_size = 32 ,batch = 'm_B',algorithm = 'Nadam',act_func = arguments.activation, intialize_method = arguments.weight_init,loss_type = arguments.loss,l2_norm = False)
object.fit(X1_train, y_train, epochs=arguments.epochs, learning_rate=arguments.learning_rate, batch_size = arguments.batch_size ,batch = 'm_B',algorithm = arguments.optimizer,act_func = arguments.activation, intialize_method = arguments.weight_init,loss_type = arguments.loss,l2_norm = False)


# In[26]:

print(object.forward_pass(X_train[0]))
Y_pred_train = object.predict(X1_train)
ys_loss = 0
system = np.array([0,0])
#print(system)
ys_loss = 0
sys = np.array([0,0])
#print(sys)
Y_pred_train = np.argmax(Y_pred_train,1)
run_sum = 0
run_sum+=1
Y_pred_val1 = object.predict(X2_train_validation)
run_sum = 0
run_sum+=1
Y_pred_val1 = np.argmax(Y_pred_val1,1)

Y_pred_val = object.predict(X_test)
ys = 0
sys = np.array([0,0])
#print(sys)
Y_pred_val = np.argmax(Y_pred_val,1)

accuracy_train = accuracy_score(Y_pred_train, Y1_train)
accuracy_val = accuracy_score(Y_pred_val1, Y2_train_validation)
accuracy_test = accuracy_score(Y_pred_val, Y_test)
print("Training accuracy", round(accuracy_train, 2))
print("Validation accuracy", round(accuracy_val, 2))
print("Test accuracy", round(accuracy_test, 2))
#wandb.log({"Training accuracys": round(accuracy_train, 2) , "Validation accuracy": round(accuracy_val, 2)})

# In[8]:



   





