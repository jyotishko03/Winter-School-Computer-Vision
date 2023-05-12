import cv2
import numpy as np

def f(x):
    pass
vid=cv2.VideoCapture(0) #Capturing live Video

#Trackbars
cv2.namedWindow('Trackbar')

cv2.createTrackbar('diameter', 'Trackbar', 15,255,f)
cv2.createTrackbar('sigma color', 'Trackbar', 69,255,f)
cv2.createTrackbar('sigma space', 'Trackbar', 103,255,f)
cv2.createTrackbar('edge control', 'Trackbar', 4,150,f)

while True:
    _,img=vid.read()
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('img',img)

#Controlling parameters of blurring
    dia = cv2.getTrackbarPos('diameter', 'Trackbar')
    s_color = cv2.getTrackbarPos('sigma color', 'Trackbar')
    s_space = cv2.getTrackbarPos('sigma space', 'Trackbar')
    blur2 = cv2.medianBlur(img, 5)
    blur2=cv2.bilateralFilter(blur2, dia, s_color, s_space)
    gaussian_mask = cv2.GaussianBlur(blur2, (11,11), 2)
    blur2 = cv2.addWeighted(blur2, 1.5, gaussian_mask, -0.5, 0)

    #cv2.imshow('blur', blur2)
    edge_c = cv2.getTrackbarPos('edge control', 'Trackbar')
    edges=cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,2*edge_c+1,2*edge_c+1)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon=cv2.bitwise_and(blur2,edges)

    cv2.imshow('cartoon', cartoon)

    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows