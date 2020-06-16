import os
import random
from random import sample
import cv2

IamgeDirs = ['graph_20200110','jpg_compress']#,'pColorImage']
f = open('train.txt','w')
f1 = open('test.txt','w')

for IamgeDir in IamgeDirs:
    files = os.listdir(IamgeDir)
    random.shuffle(files)
    tests = sample(files,int(0.2*len(files)))
    for name in files:
        if '___' not in name:
            continue

        quality = name.split('___')[0]
        oriname = name.split('___')[-1]

        dstImg = cv2.imread(os.path.join(IamgeDir,oriname))
        if dstImg is None:
            continue
        
        print(os.path.join(IamgeDir,oriname) + '  ' + quality)
        line = os.path.join(IamgeDir,oriname) + ' ' + quality + '\n'

        if name in tests:
            f1.write(line)
        else:
            f.write(line)

f.close()
f1.close()