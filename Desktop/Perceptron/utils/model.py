import numpy as np
class Perceptron:
  def __init__(self,eta,epochs,nonlinear:bool=0,earlystopping:bool=0):
    self.weights=np.random.randn(3)*1e-3
    self.eta=eta
    self.epochs=epochs
    self.nonlinear=nonlinear
    self.earlystopping=earlystopping
    print("{}Initial weights with {} errors and for {} epochs")
  def activationfunction(self,inputs,weights):
    z=np.dot(inputs,weights)
    if self.nonlinear==0:
        return np.where(z>0,1,0)
    else:
        return np.where(np.tanh(z)>0,1,0)
  def fit(self,X,Y):
    self.X=X
    self.Y=Y
    x_bias=np.c_[self.X,-np.ones((len(self.X),1))]
    c=100 # Counter to check the difference between last loss and new loss, inititated with a higher value so as the program runs
    for i in range(self.epochs):
      z=self.activationfunction(x_bias,self.weights)
      self.error=self.Y-z
      print(f"Error is: {self.error}\n","and epoch is",i)
      self.weights=self.weights+ self.eta *np.dot(x_bias.T,self.error)
      if self.earlystopping==1 and np.absolute(c-(np.sum(self.error)))<0.1: #checking whether the new loss is greater than 0.01 or not
          break  
      c=np.absolute(np.sum(self.error))
      
  def predict(self,X):
    x_bias=np.c_[X,-np.ones((len(X),1))]
    return self.activationfunction(x_bias, self.weights)
  def total_loss(self):
    print(np.sum(self.error))  