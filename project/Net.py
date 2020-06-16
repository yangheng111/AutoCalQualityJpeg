import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
        def __init__(self,n_feature,n_hidden,n_output):
            #初始网络的内部结构
            super(Net,self).__init__()
            self.hidden=torch.nn.Linear(n_feature,n_hidden)
            self.predict=torch.nn.Linear(n_hidden,n_output)
        def forward(self, x):
            #一次正向行走过程
            x=F.relu(self.hidden(x))
            x=self.predict(x)
            return x