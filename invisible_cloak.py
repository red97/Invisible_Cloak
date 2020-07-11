import numpy as np
import cv2

def nothing(x):
    pass

def setRange():
    cv2.namedWindow('Capture')
    cv2.namedWindow('Trackbars')
    
    cv2.createTrackbar('Min_H', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('Min_S', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('Min_V', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('Max_H', 'Trackbars', 0, 255, nothing)
    cv2.setTrackbarPos('Max_H', 'Trackbars', 255)
    cv2.createTrackbar('Max_S', 'Trackbars', 0, 255, nothing)
    cv2.setTrackbarPos('Max_S', 'Trackbars', 255)
    cv2.createTrackbar('Max_V', 'Trackbars', 0, 255, nothing)
    cv2.setTrackbarPos('Max_V', 'Trackbars', 255)
    
    cap = cv2.VideoCapture(0)
    
    while(True):
        ret, frame = cap.read()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        min_h = cv2.getTrackbarPos('Min_H', 'Trackbars')
        min_s = cv2.getTrackbarPos('Min_S', 'Trackbars')
        min_v = cv2.getTrackbarPos('Min_V', 'Trackbars')

        max_h = cv2.getTrackbarPos('Max_H', 'Trackbars')
        max_s = cv2.getTrackbarPos('Max_S', 'Trackbars')
        max_v = cv2.getTrackbarPos('Max_V', 'Trackbars')

        lower = np.array([min_h, min_s, min_v])
        upper = np.array([max_h, max_s, max_v])

        mask = cv2.inRange(hsv, lower, upper) 

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations = 2)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations = 1)

        blank = np.ndarray(mask.shape, np.uint8)
        blank.fill(255)
        
        cv2.imshow('Capture', mask)
        cv2.imshow('Trackbars', blank)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    return min_h, min_h, min_v, max_h, max_s, max_v

def useRange(min_h, min_s, min_v, max_h, max_s, max_v):
    cap = cv2.VideoCapture(0)
    
    background = 0 

    for i in range(60): 
        return_val, background = cap.read() 
        if return_val == False : 
            continue 

    while(True):
        ret, frame = cap.read()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower = np.array([min_h, min_s, min_v])
        upper = np.array([max_h, max_s, max_v])
        
        mask = cv2.inRange(hsv, lower, upper) 
    
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations = 2)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations = 1)
        mask1 = cv2.bitwise_not(mask)

        res1 = cv2.bitwise_and(background, background, mask = mask) 
        res2 = cv2.bitwise_and(frame, frame, mask = mask1) 
        final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
        
        cv2.imshow('Capture', final_output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    minr, ming, minb, maxr, maxg, maxb = setRange()
    useRange(minr, ming, minb, maxr, maxg, maxb)
    