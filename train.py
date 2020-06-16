import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

from Net import Net
from dataset import Dataset
import cv2


def train(DataLoader,Net,Loss,Optimizer,scheduler):
    print('------      启动训练      ------')
    for epoch in range(101):
        Net.train()
        scheduler.step()
        for i,(x,y) in enumerate(DataLoader):
            # Variable是将tensor封装了下，用于自动求导使用
            x, y = x.cuda(), y.cuda()

            Optimizer.zero_grad()  #清除上一梯度

            prediction=net(x)
            loss=Loss(prediction,y)
            # print(loss)
            
            loss.backward() #反向传播计算梯度
            optimizer.step()  #应用梯度
        
            if i%100 == 0:
                print("Train Epoch:{} iter:{} mse loss:{}".format(epoch,i,loss))

        if epoch%10 ==0:
            print("Saving")
            savepath = './model_epoch_'+str(epoch)+'.pth'
            torch.save(Net.state_dict(), savepath)
        
 
if __name__=='__main__':

    trainDataSet = Dataset('./train.txt')
    trainDataLoader = torch.utils.data.DataLoader(trainDataSet,batch_size=32,shuffle=True,num_workers=8,
                pin_memory=True,drop_last=True)
    
    print('------      搭建网络      ------')
    net = Net(n_feature=256*3,n_hidden=256,n_output=1).cuda()

    net.load_state_dict(torch.load('./20200515/model_epoch_90.pth'))
    print('网络结构为：',net)
    loss_func=F.mse_loss
    optimizer=torch.optim.SGD(net.parameters(),lr=0.01)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,50,70,90], gamma=0.1)

    train(trainDataLoader,net,loss_func,optimizer,scheduler)
