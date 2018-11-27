from scipy.cluster.vq import *
import numpy as np
from pylab import *
from PIL import Image
import imtools,pca
import pickle
from scipy.misc import imresize

steps = 50 # 图像被划分成 steps×steps 的区域
im = np.array(Image.open('empire.jpg'))
dx = im.shape[0] // steps
dy = im.shape[1] // steps
# 计算每个区域的颜色特征
features = []
for x in range(steps):
    for y in range(steps):
        R = np.mean(im[x*dx:(x+1)*dx,y*dy:(y+1)*dy,0])
        G = np.mean(im[x*dx:(x+1)*dx,y*dy:(y+1)*dy,1])
        B = np.mean(im[x*dx:(x+1)*dx,y*dy:(y+1)*dy,2])
        features.append([R,G,B])
features = array(features,'f') # 变为数组
# 聚类
centroids,variance = kmeans(features,3)
code,distance = vq(features,centroids)
# 用聚类标记创建图像
codeim = code.reshape(steps,steps)
print(codeim.shape)
codeim = imresize(codeim,im.shape[:2],interp='nearest')

figure()
imshow(codeim)
show()