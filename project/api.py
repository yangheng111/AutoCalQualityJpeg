import os
import numpy as np
import time
import cv2

import torch
from Net import Net


class AutoCalQuality():
    def __init__(self):
        super(AutoCalQuality,self).__init__()
        self.net = self.loadNet()

    def loadNet(self,path='./model_epoch_100.pth'):
        net = Net(n_feature=256*3,n_hidden=256,n_output=1)
        net.load_state_dict(torch.load(path,map_location='cpu'))
        return net

    def calHist(self,image):
        b, g, r = cv2.split(image)

        b_hist= cv2.calcHist([b], [0], None, [256], [0.0,255.0]) 
        g_hist= cv2.calcHist([g], [0], None, [256], [0.0,255.0]) 
        r_hist= cv2.calcHist([r], [0], None, [256], [0.0,255.0]) 

        norm_b_hist = (b_hist-min(b_hist))/(max(b_hist)-min(b_hist))
        norm_g_hist = (g_hist-min(g_hist))/(max(g_hist)-min(g_hist))
        norm_r_hist = (r_hist-min(r_hist))/(max(r_hist)-min(r_hist))

        hist = np.concatenate((norm_b_hist, norm_g_hist,norm_r_hist), axis=0)

        return hist

    def main(self,image):
        net = self.net

        hist = self.calHist(image)
        x = torch.from_numpy(np.array(hist).reshape(1,-1))

        prediction=net(x)
        prediction = int(prediction.data.cpu().numpy()[0][0]*100)
        if prediction > 95:
            prediction =95
        elif prediction < 30:
            prediction = 30
        
        return prediction

if __name__ == "__main__":
    imgPath = './1.jpg'
    img = cv2.imread(imgPath)

    api = AutoCalQuality()
    
    t1 = time.time()
    pred = api.main(img)
    print('using :',time.time()-t1)
    print("{}--{}".format(imgPath,pred))
        
        