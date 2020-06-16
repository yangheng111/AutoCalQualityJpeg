import os
import cv2
import numpy as np

def pca(dataMat, k):
    print(np.max(dataMat))
    average = np.mean(dataMat, axis=0) #按列求均值
    # print(average)
    m, n = np.shape(dataMat)
    # print(m,n)
    meanRemoved = dataMat - np.tile(average, (m,1)) #减去均值
    # print(meanRemoved)
    normData = meanRemoved / np.std(dataMat) #标准差归一化
    # print(normData)
    covMat = np.cov(normData.T)  #求协方差矩阵
    print(covMat)
    eigValue, eigVec = np.linalg.eig(covMat) #求协方差矩阵的特征值和特征向量
    eigValInd = np.argsort(-eigValue) #返回特征值由大到小排序的下标
    selectVec = np.matrix(eigVec.T[:k]) #因为[:k]表示前k行，因此之前需要转置处理（选择前k个大的特征值）
    finalData = normData * selectVec.T #再转置回来
    return finalData

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

if __name__ == "__main__":
    anntxts = ['train.txt','test.txt']
    
    f = open('train_feater.txt','w')
    f1 = open('test_feater.txt','w')

    for anntxt in anntxts:
        lines = open(anntxt,'r').readlines()
        for line in lines:
            imgPath = line.split()[0]
            quality = line.split()[-1]

            img = cv2.imread(imgPath)
            hist = calHist(img)

            s = ''
            for h in hist:
                s = s + str(h[0]) + ' '
            s = s + quality + '\n'
            print(s)

            if 'train' in anntxt:
                f.write(s)
            else:
                f1.write(s)
    f.close()
    f1.close()