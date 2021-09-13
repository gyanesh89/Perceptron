import pandas as pd
import numpy as np
from utils.model import Perceptron
from utils.all_utils import prepare
from utils.all_utils import saveplot
x={"x1":[1,1,0,0],"x2":[1,0,1,0],"y":[0,1,1,0]}
df=pd.DataFrame(x)
X,y =prepare(df)
model=Perceptron(eta=0.1,epochs=20,nonlinear=1,earlystopping=0)
model.fit(X,y)
model.total_loss()
saveplot(df,model,filename="xor.jpg")
