import cv2
import numpy as np
import keyboard
import operator as op

vid=cv2.VideoCapture(0)

def func(x):
    pass

color = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('color')
cv2.createTrackbar('buffer','color', 10, 255, func)
cv2.createTrackbar('red', 'color', 0, 255, func)
cv2.createTrackbar('green', 'color', 0, 255, func)
cv2.createTrackbar('blue', 'color', 0, 255, func)

flower=cv2.imread('Flower.jpg')
flower=cv2.resize(flower,(640,480))
flower_gray=cv2.cvtColor(flower,cv2.COLOR_BGR2GRAY)
kernel=(3,3)
blur=cv2.blur(flower,kernel ) # img, kernelsize
canny=cv2.Canny(blur,20,20)  # img, low, high    low=(0,255), high=(0,255)


while True:
    bf = cv2.getTrackbarPos('buffer', 'color')
    _,img=vid.read()
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = np.zeros((480, 640, 3), np.uint8)
    img2=img.copy()
    cv2.imshow('color',color)

    red = cv2.getTrackbarPos('red', 'color')
    green = cv2.getTrackbarPos('green', 'color')
    blue = cv2.getTrackbarPos('blue', 'color')

    r=img[:,:,2]
    g=img[:,:,1]
    b=img[:,:,0]



    bool_r=np.multiply((r>red-bf),(r<red+bf))
    r_i=np.where(bool_r)

    bool_g = np.multiply((g > green - bf), (g < green + bf))
    g_i = np.where(bool_g)

    bool_b = np.multiply((b > blue - bf), (b < blue + bf))
    b_i = np.where(bool_b)

    for i in range(3):
        img[r_i[0],r_i[1],i]= canny[r_i]
        img[g_i[0], g_i[1], i] = canny[g_i]
        img[b_i[0], b_i[1], i] = canny[b_i]

    cv2.imshow('img', img2)
    color[:] = [blue, green, red]
    cv2.imshow('Agency2', img)


    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
