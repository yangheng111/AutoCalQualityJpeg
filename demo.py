import os
import numpy as np
import time
import cv2

import torch
import torch.nn.functional as F

from Net import Net

def calHist(image):
    b, g, r = cv2.split(image)

    b_hist= cv2.calcHist([b], [0], None, [256], [0.0,255.0]) 
    g_hist= cv2.calcHist([g], [0], None, [256], [0.0,255.0]) 
    r_hist= cv2.calcHist([r], [0], None, [256], [0.0,255.0]) 

    norm_b_hist = (b_hist-min(b_hist))/(max(b_hist)-min(b_hist))
    norm_g_hist = (g_hist-min(g_hist))/(max(g_hist)-min(g_hist))
    norm_r_hist = (r_hist-min(r_hist))/(max(r_hist)-min(r_hist))

    hist = np.concatenate((norm_b_hist, norm_g_hist,norm_r_hist), axis=0)

    return hist
    
if __name__ == "__main__":
    net = Net(n_feature=256*3,n_hidden=256,n_output=1).cuda()
    net.load_state_dict(torch.load('./model_epoch_100.pth'))

    lines= open('./test.txt','r').readlines()
    error =0
    n = 0
    f= open('./file.txt','w')
    for line in lines:
        name = line.split()[0]
        quality = line.split()[-1]

        imgPath = name#os.path.join('./testImg',name)
        img = cv2.imread(imgPath)

        hist = calHist(img)
        x = torch.from_numpy(np.array(hist).reshape(1,-1)).cuda()

        prediction=net(x)
        prediction = int(prediction.data.cpu().numpy()[0][0]*100)
        if prediction > 95:
            prediction =95
        elif prediction < 30:
            prediction = 30

        s = './jpeg-recompress '+name.split('/')[-1]+' '+ str(prediction) + '\n'
        f.write(s)
        # n = n+1
        # if abs(int(prediction.data.cpu().numpy()[0][0]*100)-int(quality))>10:
        #     error =error+1
        #     print('imgPath :',imgPath)
        #     print('error:{},right:{},all:{},predict:{},label:{}'.format(error,n,len(lines),int(prediction.data.cpu().numpy()[0][0]*100),quality))