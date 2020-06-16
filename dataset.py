import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

import cv2

def calHist(image):
    b, g, r = cv2.split(image)
    b_hist= cv2.calcHist([b], [0], None, [256], [0.0,255.0]) 
    g_hist= cv2.calcHist([g], [0], None, [256], [0.0,255.0]) 
    r_hist= cv2.calcHist([r], [0], None, [256], [0.0,255.0]) 
    # print("ori:",b_hist)
    # print(max(b_hist),min(b_hist))
    norm_b_hist = (b_hist-min(b_hist))/(max(b_hist)-min(b_hist))
    norm_g_hist = (g_hist-min(g_hist))/(max(g_hist)-min(g_hist))
    norm_r_hist = (r_hist-min(r_hist))/(max(r_hist)-min(r_hist))
    # print("dat",norm_b_hist)
    hist = np.concatenate((norm_b_hist, norm_g_hist,norm_r_hist), axis=0)
    return hist

class Dataset(Dataset):
    def __init__(self,anntxt):
        ann = open(anntxt,'r')
        self.lines = ann.readlines()

    def __getitem__(self, index):
        line = self.lines[index]

        imgPath = line.split()[0]
        quality = float(line.split()[-1])/100

        img = cv2.imread(imgPath)

        hist = torch.from_numpy(np.array(calHist(img)).reshape(1,-1))
        label = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(np.array(quality,dtype=float)).float(),dim=0),dim=0)

        return hist, label


    def __len__(self):
        return len(self.lines)

if __name__ == "__main__":
    trainDataSet = Dataset('./train.txt')
    trainDataLoader = torch.utils.data.DataLoader(trainDataSet,batch_size=1,shuffle=True,num_workers=1,
                pin_memory=False,drop_last=True)

    for i,(x,y) in enumerate(trainDataLoader):
        print(i)