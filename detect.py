import cv2
import numpy as np
import time 
from PIL import Image



cap = cv2.VideoCapture("input2.mp4")

LOWER = np.array([20, 100, 50])
UPPER = np.array([35, 255, 255])

kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],
                                      [0,1,0,0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],
                                     [0,1,0,1],
                                     [0,0,1,0],
                                     [0,0,0,1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.7

while True:
    ret, frame = cap.read()
    time.sleep(0.1)
    
    if not ret:
        print("Error: Could not read frame or video ended.")
        break

    img =  cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img,LOWER,UPPER)
    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)


        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        kalman.correct(measurement)

        
        predicted = kalman.predict()
        px, py = int(predicted[0]), int(predicted[1])

        cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 2)
        cv2.circle(frame, (px, py), 15, (255, 0, 0), 2)
        # f_future = kalman.statePost.copy()

        # for i in range(5):
        #     f_future = np.dot(kalman.transitionMatrix,f_future)
        #     px, py = int(f_future[0]),int(f_future[1])
        #     cv2.circle(frame,(px,py),10,(255,0,0),-1)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()