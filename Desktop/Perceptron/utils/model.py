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
        return np.where(np.tan(z)>0,1,0)
  def fit(self,X,Y):
    self.X=X
    self.Y=Y
    x_bias=np.c_[self.X,-np.ones((len(self.X),1))]
    c=100
    for i in range(self.epochs):
      z=self.activationfunction(x_bias,self.weights)
      self.loss=self.Y-z
      print(f"Error is: {self.loss}\n","and epoch is",i)
      self.weights=self.weights+ self.eta *np.dot(x_bias.T,self.loss)
      if self.earlystopping==1 and np.absolute(c-(np.sum(self.loss)))<0.1:
          break  
      c=np.absolute(np.sum(self.loss))
      
  def predict(self,X):
    x_bias=np.c_[X,-np.ones((len(X),1))]
    return self.activationfunction(x_bias, self.weights)
  def total_loss(self):
    print(np.sum(self.loss))  