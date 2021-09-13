from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def prepare(df):
  x=df.drop("y",axis=1)
  y=df["y"]
  return x,y
def saveplot(df,model):
  def _scatter(df):
    df.plot(kind="scatter",x="x1",y="x2",c="y",s=100,cmap="summer")
  def _advancedgraph (X,Y, classifier,resolution=1):
    colors = ("gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(Y))])
    x=X.values
    x1=x[:,0]
    x2=x[:,0]
    x1min,x1max=x1.min()-2,x1.max()+2
    x2min,x2max=x2.min()-2,x2.max()+2
    xx1,xx2=np.meshgrid(np.arange(x1min,x1max,resolution),np.arange(x2min,x2max,resolution))
    z=classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z=z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.2,cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.plot()
    plt.savefig("and.png")
  X,Y=prepare(df)
  _scatter(df)
  _advancedgraph(X, Y, model)
