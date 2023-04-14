import cv2
import numpy as np

#cv2.namedWindow('Live Stream')
vid=cv2.VideoCapture(r"C:\Users\jyoti\Desktop\Curvy Road.mp4")

while True:
    _,img=vid.read()
    cv2.imshow('Road', img)

    if cv2.waitKey(27) & 0xFF==ord('q'):
        break
    crop=img[img.shape[0]//4+100:, img.shape[1]//8:3*img.shape[1]//4,:]
    crop = img[img.shape[0] // 4 + 300:, :, :]
    imgray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    ret, thresh=cv2.threshold(imgray,135,255,0)
#cv2.imshow(ret)

    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print('Number of contours='+str(len(contours)))


    cv2.drawContours(crop,contours,-1,(0,255,0),2)



    cv2.namedWindow('InputImg', cv2.WINDOW_NORMAL)
    cv2.imshow('InputImg', img)
    #cv2.imshow('Image GRAY', imgray)










vid.release()
cv2.destroyAllWindows