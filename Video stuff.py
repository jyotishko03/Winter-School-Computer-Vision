import cv2
import numpy as np

vid=cv2.VideoCapture(0)

#Creating a Trackbar
def func(x):
    pass




# Creating a window with black image
color = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('color')


cv2.createTrackbar('red', 'color', 0, 255, func)
cv2.createTrackbar('green', 'color', 0, 255, func)
cv2.createTrackbar('blue', 'color', 0, 255, func)
cv2.createTrackbar('threashold','color', 0, 255, func)


flower=cv2.imread('Flower.jpg')
flower=cv2.resize(flower,(640,480))
flower_gray=cv2.cvtColor(flower,cv2.COLOR_BGR2GRAY)
kernel=(3,3)
blur=cv2.blur(flower,kernel ) # img, kernelsize
canny=cv2.Canny(blur,20,20)  # img, low, high    low=(0,255), high=(0,255)

while True:
    _,img=vid.read()
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = np.zeros((480, 640, 3), np.uint8)
    cv2.imshow('img',img)
    cv2.imshow('color',color)


    red = cv2.getTrackbarPos('red', 'color')
    green = cv2.getTrackbarPos('green', 'color')
    blue = cv2.getTrackbarPos('blue', 'color')
    bf=cv2.getTrackbarPos('threashold', 'color')



    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if (img[i,j,2]>red-bf and img[i,j,2]<red+bf):  #Condition to zone out
                if (img[i,j,1]>green-bf and img[i,j,1]<green+bf):
                    if (img[i,j,0]>blue-bf and img[i,j,0]<blue+bf):
                        for k in range(3):
                            img[i,j,k]=0
                            box[i,j,k]=255
    box_not=cv2.bitwise_not(box)
    box_bin=cv2.cvtColor(box,cv2.COLOR_BGR2GRAY)
    flower_merged1=cv2.bitwise_and(canny,box_bin)
    back_template= np.zeros((480, 640, 3), np.uint8)
    for i in range(back_template.shape[0]):
        for j in range(back_template.shape[1]):
           for k in range(3):
               back_template[i,j,k]=flower_merged1[i,j]

    final=cv2.bitwise_or(back_template,img)
    color[:] = [blue, green, red]
    '''cv2.imshow('Agency', img)
    cv2.imshow('Agency2', box)
    cv2.imshow('Agency3', box_not)
    cv2.imshow('flower',flower)
    cv2.imshow('canny', canny)
    cv2.imshow('flower_merged1', flower_merged1)'''
    cv2.imshow('final', final)
    if cv2.waitKey(7) == ord('q'):
        break
cv2.destroyAllWindows()
