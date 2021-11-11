import cv2
import os
import numpy as np
from anpr.detect_box_yolo import detect_box

#Defining a class to extract the licence plate
class extraction:

    def __init__(self,img_path,name,threshold):
        self.img_path=img_path
        self.name=name
        self.threshold=threshold

    def read_img(self):
        color=(128,0,128)
        img= cv2.imread(self.img_path)
        rois = detect_box(img, self.threshold)
        for roi in rois:
            if roi[4] == 0:
                temp=img[roi[1]:roi[3],roi[0]:roi[2]]
        out_path=self.img_path.replace(self.name,'')
        cv2.imwrite(out_path+'\\temp_'+self.name,temp)
        return (out_path+'\\temp_'+self.name)
