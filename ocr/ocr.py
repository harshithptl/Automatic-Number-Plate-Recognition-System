import cv2
import numpy as np
from statistics import mode
from math import sqrt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#Helper functions
def return_contours(contours):
    total_contours=[]
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        total_contours.append((x,y,w,h))
    return total_contours

def dist_between_points(p1,p2):
    x1=p1[0]
    y1=p1[1]
    x2=p2[0]
    y2=p2[1]
    return sqrt(((x2-x1)**2)+((y2-y1)**2))


def character(in_path):
    #All the classes
    categories=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

    #Taking the input from the Licence Plate localization pipeline and pre-processing
    img=cv2.imread(in_path)
    img=cv2.resize(img,(720,240))
    blur = cv2.bilateralFilter(img.copy(),8,75,75)
    _, thresh = cv2.threshold(blur.copy(), 100, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh_1 = cv2.threshold(gray, 100, 255, 0)

    #Obtaining all the raw contours
    contours, hierarchy = cv2.findContours(thresh_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours=[]

    #Filtering the contours based on aspect ratio(In general aspect ratio of the characters are less than 1)
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour)>750 and cv2.contourArea(contour)<10000:
            aspectRatio = w / float(h)
            if aspectRatio<1.0:
                filtered_contours.append(contour)


    #Removing the contours that are formed inside other contours
    #Useful when inconsistencies are detected inside the characters recognized
    del_ind=[]
    for ind,c in enumerate(filtered_contours):
        (x,y,w,h) = cv2.boundingRect(c)
        for ind1,c1 in enumerate(filtered_contours):
            if ind!=ind1:
                (x1,y1,w1,h1) = cv2.boundingRect(c1)
                if x<x1 and x1<x+w and y<y1 and y1<y+h and x<x1+w and x1<x+w and y<y1+h1 and y1+h1<y+h:
                    if cv2.contourArea(c)>cv2.contourArea(c1):
                        del_ind.append(ind1)
                    else:

                        del_ind.append(ind)
    contour_final=[con for ind,con in enumerate(filtered_contours) if ind not in del_ind]



    filtered_contours=return_contours(contour_final)
    filtered_contours=sorted(filtered_contours, key=lambda x: x[0])

    #Algorithm to filter out outliers
    #The distance between top left and bottom right points are calculated and appended to separate lists
    #Then the Z-value of each observation is calculated
    #If the Z-value is greater than some threshold it is an outlier
    #The loop breaks if total contours is 10 or there are no outliers in both mentioned lists
    while(True):
        count=0
        top_left=[]
        bottom_right=[]

        for ind,i in enumerate(filtered_contours):
            x=i[0]
            y=i[1]
            w=i[2]
            h=i[3]
            x1=filtered_contours[ind+1][0]
            y1=filtered_contours[ind+1][1]
            w1=filtered_contours[ind+1][2]
            h1=filtered_contours[ind+1][3]
            top_left.append((abs(x1-x)**2+abs(y1-y)**2)**0.5)
            bottom_right.append((abs((x1+w1)-(x+w))**2+abs((y1+h1)-(y+h))**2)**0.5)
            if ind==len(filtered_contours)-2:
                break

        mean_x = np.mean(top_left)
        std_x = np.std(top_left)
        mean_y = np.mean(bottom_right)
        std_y = np.std(bottom_right)

        #Top left
        threshold = 1.25
        outlier_1 = []
        for ind,i in enumerate(top_left):
            z = (i-mean_x)/std_x
            if z > threshold:
                outlier_1.append(ind)

        if len(outlier_1)==0:
            count+=1
        else:
            for i in outlier_1:
                if i<1:
                    filtered_contours=filtered_contours[i+1:]
                elif i>=9:
                    filtered_contours=filtered_contours[0:i+1]

        #Bottom right
        threshold = 1.25
        outlier_2 = []
        for ind,i in enumerate(bottom_right):
            z = (i-mean_y)/std_y
            if z > threshold:
                outlier_2.append(ind)
        if len(outlier_2)==0:
            count+=1
        else:
            for i in outlier_2:
                if i<1 and i not in outlier_1:
                    filtered_contours=filtered_contours[i+1:]
                elif i>=9 and i not in outlier_1:
                    filtered_contours=filtered_contours[0:i+1]


        if len(filtered_contours)==10:
            break
        if count==2:
            break


    contour_final=filtered_contours


    for contour in contour_final:
        x = contour[0]
        y = contour[1]
        w = contour[2]
        h = contour[3]
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    #Loading model for predictions
    out=[]
    model=load_model('model.h5')

    #Performing similar pre-processing as our training data
    for contour in contour_final:
        x = contour[0]
        y = contour[1]
        w = contour[2]
        h = contour[3]
        test=thresh_1[y:y+h,x:x+w]
        test=cv2.resize(test,(64,64))
        test=(image.img_to_array(test))/255
        test=np.expand_dims(test, axis = 0)
        result=model.predict(test)
        out.append(categories[np.argmax(np.array(result))])


    #Printing the output
    #Decided against segmentation of the final characters as my OCR system isn't perfect
    plate=''.join(out)
    return plate
