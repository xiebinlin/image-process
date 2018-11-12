from PIL import Image
import os
from numpy import *
from pylab import *

def normalize(points):
    for row in points:
        row /= points[-1]
    return points
 
def make_homog(points):
    return vstack((points,ones((1,points.shape[1]))))
    
def H_from_points(fp,tp): 
    """ 使用线性 DLT 方法，计算单应性矩阵 H，使 fp 映射到 tp。点自动进行归一化 """
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
    # 对点进行归一化（对数值计算很重要）
    
    # --- 映射起始点 ---
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = dot(C1,fp)
    
    # --- 映射对应点 ---
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 = diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = dot(C2,tp)
    
    # 创建用于线性方法的矩阵，对于每个对应对，在矩阵中会出现两行数值
    nbr_correspondences = fp.shape[1]
    A = zeros((2*nbr_correspondences,9))
    for i in range(nbr_correspondences):
        A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
        A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]
    U,S,V = linalg.svd(A)
    H = V[8].reshape((3,3))
    # 反归一化
    H = dot(linalg.inv(C2),dot(H,C1))
    # 归一化，然后返回
    return H / H[2,2]
    
def Haffine_from_points(fp,tp):
    """ 计算 H，仿射变换，使得 tp 是 fp 经过仿射变换 H 得到的 """ 
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
    # 对点进行归一化
    
    # --- 映射起始点 ---
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = dot(C1,fp)
    
    # --- 映射对应点 ---
    m = mean(tp[:2], axis=1)
    C2 = C1.copy() # 两个点集，必须都进行相同的缩放
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = dot(C2,tp)
    
    # 因为归一化后点的均值为 0，所以平移量为 0
    A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
    U,S,V = linalg.svd(A.T)
    # 如 Hartley 和 Zisserman 著的Multiple View Geometry in Computer, Scond Edition 所示，
    # 创建矩阵 B 和 C
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]
    tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1)
    
    H = vstack((tmp2,[0,0,1]))
    # 反归一化
    H = dot(linalg.inv(C2),dot(H,C1))
    return H / H[2,2]
    
    