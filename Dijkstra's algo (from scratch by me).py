import cv2
import numpy as np
import keyboard

#Defining colors
BLACK=[0,0,0]
WHITE=[255,255,255]
RED=[0,0,255]
GREEN=[0,255,0]
CYAN=[255,255,0]
BLUE=[255,0,0]
YELLOW=[0,255,255]

map=cv2.imread("processed_img.png")
global ret
ret=[]
start=(156,435)
#end=(163,88)
#end=(12, 546)
end=(17,395)
end=(100,492)
#end=(170,430)


#for getting start and end according to user's choice
def start_end_track(matrix2):
    matrix=matrix2.copy()

    steps=1
    center = [matrix.shape[0] // 2, matrix.shape[1] // 2]
    flag=0
    while True:
        display=np.zeros((100,500,3), dtype=np.uint8)
        img_track=matrix.copy()
        cv2.putText(display, "use w,a,s,d to move pointer and set it On Path (white)", [12, 10], cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    YELLOW, 1,
                    cv2.LINE_AA)

        if keyboard.is_pressed('w'):
            center[0] -= steps
        elif keyboard.is_pressed('s'):
            center[0] += steps
        elif keyboard.is_pressed('a'):
            center[1] -= steps
        elif keyboard.is_pressed('d'):
            center[1] += steps

        cv2.putText(display, 'x: '+str(center[1])+', y: '+str(center[0]), [20,80], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1,
                    cv2.LINE_AA)
        matrix_gray=cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
        if flag==0:
            cv2.putText(display, "To fix Start point, press 'k' ", [200, 40], cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, RED, 1,
                    cv2.LINE_AA)
        elif flag==1:
            cv2.putText(display, "To fix End point press 'k' ", [200, 40], cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, GREEN, 1,
                    cv2.LINE_AA)
        cv2.putText(display, "after bringing the pointer on path", [210, 70], cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, YELLOW, 1,
                    cv2.LINE_AA)
        if matrix_gray[center[0], center[1]]==255:
            cv2.putText(display, 'On Path', [20, 40], cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, GREEN, 1,
                        cv2.LINE_AA)
        else:
            cv2.putText(display, 'Not On Path!', [20, 40], cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, RED, 1,
                        cv2.LINE_AA)

        if keyboard.is_pressed('k'):
            if flag==0:
                center2=center.copy()
                ret.append(center2)
                cv2.putText(display, 'Start allocated!!', [300, 90], cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, CYAN, 1,
                        cv2.LINE_AA)
                flag=1
                cv2.imshow('display', display)

                cv2.waitKey(1000)
                continue
            elif flag==1:
                center3=center.copy()
                ret.append(center3)
                cv2.putText(display, 'End allocated!!', [300, 90], cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, CYAN, 1,
                            cv2.LINE_AA)
                cv2.imshow('display', display)

                cv2.waitKey(1000)
                break



        if flag==0:
            cv2.circle(img_track,[center[1],center[0]], 3, RED, -1)
        elif flag==1:
            cv2.circle(img_track, [center[1], center[0]], 3, GREEN, -1)

        img_track2 = cv2.resize(img_track, (int(img_track.shape[1] * 1.5), int(img_track.shape[0] * 1.5)), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('tracker', img_track2)
        cv2.imshow('display', display)
        cv2.waitKey(1)




def calcDist(point,current):
    return ((point[0]-current[0])**2+(point[1]-current[1])**2)**(1/2)

def inImg(img,point): #Checks if point lies inside img
    return ((point[0]>=0 and point[0]<img.shape[0]) and (point[1]>=0 and point[1]<img.shape[1]))

def display_Path(map,start,end,current,parent):
    map_path=map.copy()
    pxl=current
    while current != start:
        pxl = int(parent[current][0]), int(parent[current][1])
        map_path[int(pxl[0]), int(pxl[1])] = BLUE
        current = (pxl[0], pxl[1])


    cv2.circle(map_path, (start[1],start[0]), 3, RED, -1)
    cv2.circle(map_path, (end[1], end[0]), 3, GREEN, -1)
    map_path2 = cv2.resize(map_path, (map.shape[1]*2, map.shape[0]*2), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('map',map_path2)
    cv2.waitKey(1)


def Dijktra(map, start, end):
    img=map.copy()
    h,w,t=img.shape
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Dist_from_start=np.full((h,w), np.inf) #Initially, all points are at infinity wrt end
    Dist_from_start[start]=0
    parent=np.zeros((h,w,2)) #each element stores co-ordinate of parent pixel
    visited=np.full((h,w), False)

    current=start

    while current!=end:
        visited[current]=True
        #print(current)

        for i in range(-1,2):
            for j in range(-1,2):

                point=(current[0]+i,current[1]+j)
                #print(point)
                if inImg(img, point):
                    if (map[point]==WHITE).all():
                        if (calcDist(point, current)+Dist_from_start[current]< Dist_from_start[point]):
                            Dist_from_start[point]=calcDist(point, current)+Dist_from_start[current]
                            parent[point[0], point[1]]=[current[0], current[1]]

        min_dist = np.inf
        #This finds out among all points whose distances has been assigned, which is closest to start
        '''
        for i in range(h):
            for j in range(w):
                if Dist_from_start[i, j] < min_dist:
                    if visited[i, j]==False:
                        min_dist = Dist_from_start[i, j]
                        current = (i, j)
                        print(current)
        
        print(current)'''

        #Although the above commented snippet is more understandable
        #But the below snippet does the same job but with much efficiency

        hypothetical = Dist_from_start.copy()
        n=np.where(visited==True)
        hypothetical[n[0], n[1]] = np.inf #assigned pixels already visited to infinity
        p=np.where(hypothetical==hypothetical.min()) #the above step, cleverly lets us get the min distance of unexplored point

        min_dist=hypothetical.max()
        current=(int(p[0][len(p[0])-1]),int(p[1][len(p[0])-1]))

        display_Path(map, start, end, current, parent) #during path finding process

    display_Path(map, start, end, current, parent) #final path display
    cv2.waitKey(0)




reply=input("Do you wish to set Start and End Point yourself (y/n)?")
if reply=='y':
    print("Please check taskbar for 2 extra windows")
    start_end_track(map)
    print("Please check taskbar for 2 extra windows")
    cv2.destroyAllWindows()

    start = (ret[0][0], ret[0][1])
    end = (ret[1][0], ret[1][1])

    if (map[start] == BLACK).all() or (map[end] == BLACK).all():
        print("Start and End points are not on path!!!\nRun again")
    else:
        Dijktra(map, start, end)
    cv2.destroyAllWindows()

elif reply=='n':
    print("Using default Points....")
    cv2.waitKey(1000)
    print("Please check taskbar for an extra window")
    Dijktra(map, start, end)
else:
    print("Wrong Reply. Rerun")





