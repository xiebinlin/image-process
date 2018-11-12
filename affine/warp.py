from numpy import *
import homography
from scipy import ndimage
import matplotlib.delaunay as md
from PIL import Image
from pylab import *

def image_in_image(im1,im2,tp):
    """ 使用仿射变换将 im1 放置在 im2 上，使 im1 图像的角和 tp 尽可能的靠近
    tp 是齐次表示的，并且是按照从左上角逆时针计算的 """
    
    # 扭曲的点
    m,n = im1.shape[:2]
    fp = array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])
    
    # 计算仿射变换，并且将其应用于图像 im1
    H = homography.Haffine_from_points(tp,fp)
    im1_t = ndimage.affine_transform(im1,H[:2,:2],(H[0,2],H[1,2]),im2.shape[:2])
    alpha = (im1_t > 0)
    
    return (1-alpha)*im2 + alpha*im1_t
    
def triangulate_points(x,y):
    """ 二维点的 Delaunay 三角剖分 """
    
    centers,edges,tri,neighbors = md.delaunay(x,y)
    return tri
    
def alpha_for_triangle(points,m,n):
    """ 对于带有由 points 定义角点的三角形，创建大小为 (m，n) 的 alpha 图
    （在归一化的齐次坐标意义下）"""
    alpha = zeros((m,n))
    for i in range(int(min(points[0])),int(max(points[0]))): 
        for j in range(int(min(points[1])),int(max(points[1]))):
            x = linalg.solve(points,[i,j,1])
            if min(x) > 0: # 所有系数都大于零
                alpha[i,j] = 1
    return alpha
  
def pw_affine(fromim,toim,fp,tp,tri):
    """ 从一幅图像中扭曲矩形图像块
    fromim= 将要扭曲的图像
    toim= 目标图像
    fp= 齐次坐标表示下，扭曲前的点
    tp= 齐次坐标表示下，扭曲后的点
    tri= 三角剖分 """
    im = toim.copy()
    
    # 检查图像是灰度图像还是彩色图象
    is_color = len(fromim.shape) == 3
    
    # 创建扭曲后的图像（如果需要对彩色图像的每个颜色通道进行迭代操作，那么有必要这样做）
    im_t = zeros(im.shape, 'uint8')
    
    for t in tri:
        # 计算仿射变换
        H = homography.Haffine_from_points(tp[:,t],fp[:,t])
        if is_color:
            for col in range(fromim.shape[2]):
                im_t[:,:,col] = ndimage.affine_transform(fromim[:,:,col],H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])
        else:
            im_t = ndimage.affine_transform(fromim,H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])
    
        # 三角形的 alpha
        alpha = alpha_for_triangle(tp[:,t],im.shape[0],im.shape[1])
  
        # 将三角形加入到图像中
        im[alpha>0] = im_t[alpha>0]
    
    return im
 
def plot_mesh(x,y,tri):
    """ 绘制三角形 """ 
    for t in tri:
        t_ext = [t[0], t[1], t[2], t[0]] # 将第一个点加入到最后
        plot(x[t_ext],y[t_ext],'r')