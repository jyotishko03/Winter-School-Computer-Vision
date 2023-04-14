import cv2
import numpy as np

def func(x):
    pass
cv2.namedWindow('Trackbar')
cv2.createTrackbar('threashold','Trackbar',0,255, func)
cv2.createTrackbar('min','Trackbar',0,255, func)
cv2.createTrackbar('max','Trackbar',0,255, func)
vid=cv2.VideoCapture(r"C:\Users\jyoti\Desktop\Smooth road.mp4")


while True:
    _,img=vid.read()
    cv2.imshow('Road', img)

    if cv2.waitKey(27) & 0xFF==ord('q'):
        break
    #crop=img[img.shape[0]//4+100:, img.shape[1]//8:3*img.shape[1]//4,:]
    crop = img[img.shape[0] // 4 + 90:, :3 * img.shape[1] // 4, :]
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)
    canny=cv2.Canny(blur,50,130)
    contours,temp=cv2.findContours(canny,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #print('Number of contours='+str(len(contours)))

    threashold=cv2.getTrackbarPos('threashold','Trackbar')
    min = cv2.getTrackbarPos('min', 'Trackbar')
    max = cv2.getTrackbarPos('max', 'Trackbar')

    mask=np.zeros(crop.shape[:2], dtype = np.uint8)
    for cnt in contours:
        x,y, width, height = cv2.boundingRect(cnt) #Used for making effective mask
        if width*height>4000:
            mask=cv2.rectangle(mask,(x,y),(x+width,y+height),255,-1)

    masked=cv2.bitwise_and(canny, canny, mask=mask)
    lines=cv2.HoughLinesP(masked,1,np.pi/180,threashold, minLineLength=min, maxLineGap=max)
    print(lines)
    try:
        for points in lines:
            x1,y1,x2,y2=points[0]
            cv2.line(crop,(x1,y1),(x2,y2),(0,255,0),2)
    except:
        pass





    cv2.namedWindow('InputImg', cv2.WINDOW_NORMAL)
    cv2.imshow('InputImg', img)
    #cv2.imshow('Image GRAY', imgray)


vid.release()
cv2.destroyAllWindows