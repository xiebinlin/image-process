import cv2

camera = cv2.VideoCapture("111.avi") # 参数0表示第一个摄像头
mog = cv2.createBackgroundSubtractorMOG2(history = 500,detectShadows=True)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while (1):
    grabbed, frame_lwpCV = camera.read()
    fgmask = mog.apply(frame_lwpCV)
    th = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
    
    dilated = cv2.dilate(th, es, iterations=10)
    image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
    for c in contours:
        if cv2.contourArea(c) > 2000:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
    
    
    cv2.imshow('frame', fgmask)
    cv2.imshow('thresh', th)
    cv2.imshow('detection', frame_lwpCV)
    key = cv2.waitKey(10) & 0xFF
    # 按'q'健退出循环
    if key == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()





# # coding:utf8
# import cv2
# def detect_video(video):    
    # camera = cv2.VideoCapture(video)    
    # history = 20    # 训练帧数    
    # bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # 背景减除器，设置阴影检测  
    # bs.setHistory(history)    
    # frames = 0    
    
    # while True:  
        # res, frame = camera.read()        
        # if not res:            
            # break
        
        # fg_mask = bs.apply(frame)   # 获取 foreground mask        
        # if frames < history:            
            # frames += 1            
            # continue        
        # # 对原始帧进行膨胀去噪        
        # th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]        
        # th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)      
        # dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)        # 获取所有检测框        
        # image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
        # for c in contours:            
        # # 获取矩形框边界坐标            
            # x, y, w, h = cv2.boundingRect(c)            
        # # 计算矩形框的面积            
            # area = cv2.contourArea(c)            
        # if 500 < area < 3000:                
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)        
            # cv2.imshow("detection", frame)        
            # cv2.imshow("back", dilated)       
        # if cv2.waitKey(110) & 0xff == 27:            
            # break    
    
    # camera.release()
    
# if __name__ == '__main__':
    # video = '111.avi'    
    # detect_video(video)