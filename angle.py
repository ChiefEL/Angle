import cv2
import numpy
import math

path = "C:/Users/Admin/Desktop/angle/dist.jpg"

img1 = cv2.imread(path)
pointsList = []

def mousePoints(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img1,(x,y),5,(0,0,255),cv2.FILLED)
        pointsList.append([x,y])
        print(pointsList)

def gradiant(pt1,pt2):
    return(pt2[1] - pt1[1])/(pt2[0]-pt1[0])

def getAngle(pointsList):
    pt1 = pt2 = pt3 = pointsList[-3:]
    #print(pt1, pt2, pt3)
    m1 = gradiant(pt1,pt2)
    m2 = gradiant(pt2,pt3)
    angR = math.atan((m2-m1)/(1+(m2*m1)))
    angleD = round(math.degrees(angR))
    if angleD < 0:
       angD_D = angleD + 180
    elif angleD > 0 :
        angD_D = angleD
    print(angD_D)


while True:

    if len(pointsList) % 3 == 0 and len(pointsList) != 0:
        getAngle(pointsList)
        cv2.imshow('image', img1)
        cv2.setMouseCallback('image', mousePoints)
    if cv2.waitKey(1) & 0xff == ord('q'):
       pointList = []
       img1 = cv2.imread(path)

