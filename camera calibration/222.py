import homography
import sift
import camera
from pylab import *
from PIL import Image

def cube_points(c,wid):
    """ 创建用于绘制立方体的一个点列表（前 5 个点是底部的正方形，一些边重合了）""" 
    p = []
    # bottom
    p.append([c[0]-wid, c[1]-wid, c[2]-wid])
    p.append([c[0]-wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]-wid, c[2]-wid])
    p.append([c[0]-wid, c[1]-wid, c[2]-wid]) #same as first to close plot
    
    # top
    p.append([c[0]-wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]-wid, c[2]+wid]) #same as first to close plot
    
    # vertical sides
    p.append([c[0]-wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]-wid])
    
    return array(p).T
    
def my_calibration(sz):
    """
    Calibration function for the camera (iPhone4) used in this example.
    """
    row, col = sz
    fx = 2555*col/2592
    fy = 2586*row/1936
    K = diag([fx, fy, 1])
    K[0, 2] = 0.5*col
    K[1, 2] = 0.5*row
    return K

sift.process_image('book_frontal.JPG','im0.sift')
l0,d0 = sift.read_features_from_file('im0.sift')
sift.process_image('book_perspective.JPG','im1.sift')
l1,d1 = sift.read_features_from_file('im1.sift')

# 匹配特征，并计算单应性矩阵

# match features and estimate homography
matches = sift.match_twosided(d0, d1)  #匹配
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx, :2].T)  #vstack
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2, :2].T) #vstack

model = homography.RansacModel()
H, inliers = homography.H_from_ransac(fp, tp, model) #删除误匹配

# camera calibration
K = my_calibration((747, 1000))   #相机焦距光圈初始化

# 3D points at plane z=0 with sides of length 0.2
box = cube_points([0, 0, 0.1], 0.1)  

# project bottom square in first image
cam1 = camera.Camera(hstack((K, dot(K, array([[0], [0], [-1]]))))) #相机初始化
# first points are the bottom square
box_cam1 = cam1.project(homography.make_homog(box[:, :5]))   #在相机中的坐标


# use H to transfer points to the second image
box_trans = homography.normalize(dot(H,box_cam1)) #单应性矩阵的变换

# compute second camera matrix from cam1 and H
cam2 = camera.Camera(dot(H, cam1.P))   #相机的外参数
A = dot(linalg.inv(K), cam2.P[:, :3])
A = array([A[:, 0], A[:, 1], cross(A[:, 0], A[:, 1])]).T
cam2.P[:, :3] = dot(K, A)   #照相机矩阵的变换

# project with the second camera
box_cam2 = cam2.project(homography.make_homog(box))  #得到cube在第二幅图中的形状 通过照相机矩阵的变换
# plotting
im0 = array(Image.open('book_frontal.JPG'))
im1 = array(Image.open('book_perspective.JPG'))

figure()
imshow(im0)
plot(box_cam1[0, :], box_cam1[1, :], linewidth=3)
title('2D projection of bottom square')
axis('off')

figure()
imshow(im1)
plot(box_trans[0, :], box_trans[1, :], linewidth=3)
title('2D projection transfered with H')
axis('off')

figure()
imshow(im1)
plot(box_cam2[0, :], box_cam2[1, :], linewidth=3)
title('3D points projected in second image')
axis('off')

show()

import pickle
with open('ar_camera.pkl','wb') as f:
    pickle.dump(K,f) 
    pickle.dump(dot(linalg.inv(K),cam2.P),f)
