from scipy.cluster.vq import *
import numpy as np
from pylab import *
from PIL import Image
import imtools,pca
import pickle

# 获取 selected-fontimages 文件下图像文件名，并保存在列表中
imlist = imtools.get_imlist('C:\\Users\\msi\\Desktop\\selectedfontimages\\a_selected_thumbs\\')

imnbr = len(imlist)
# 载入模型文件

# 创建矩阵，存储所有拉成一组形式后的图像
immatrix = np.array([np.array(Image.open(im)).flatten() for im in imlist],'f')
V, S, immean = pca.pca(immatrix)

# 投影到前 40 个主成分上
projected = np.array([np.dot(V[[0,1]],immatrix[i]-immean) for i in range(imnbr)])

# projected = whiten(projected)
# centroids,distortion = kmeans(projected,2)
# code,distance = vq(projected,centroids)

# for k in range(2):
    # ind = where(code==k)[0]
    # figure()
    # gray()
    # for i in range(minimum(len(ind),40)):
        # subplot(4,10,i+1)
        # imshow(immatrix[ind[i]].reshape((25,25)))
        # axis('off')
        

from PIL import Image, ImageDraw
# 高和宽
h,w = 1200,1200
# 创建一幅白色背景图
img = Image.new('RGB',(w,h),(255,255,255))
draw = ImageDraw.Draw(img)

# 绘制坐标轴
draw.line((0,h/2,w,h/2),fill=(255,0,0))
draw.line((w/2,0,w/2,h),fill=(255,0,0))

# 缩放以适应坐标系
scale = abs(projected).max(0)

scaled = floor(array([(p/scale) * (w/2 - 20, h/2 - 20) + (w/2, h/2) for p in projected])).astype(int)
# 粘贴每幅图像的缩略图到白色背景图片

for i in range(imnbr):
  nodeim = Image.open(imlist[i])
  nodeim.thumbnail((25, 25))
  ns = nodeim.size
  box = (scaled[i][0] - ns[0] // 2, scaled[i][1] - ns[1] // 2,scaled[i][0] + ns[0] // 2 + 1, scaled[i][1] + ns[1] // 2 + 1)
  img.paste(nodeim, box)

img.show()