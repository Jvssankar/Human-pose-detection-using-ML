import cv2
import numpy as np
import matplotlib.pyplot as plt


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


width = 368
height = 368
inWidth = width
inHeight = height

net = cv2.dnn.readNetFromTensorflow("out.pb")

thres = 0.2

cap = cv2.VideoCapture("dance3.mp4")

def pose_estimation(cap):
    
    
    
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        frame = cv2.resize(frame,(0,0),fx = 0.5,fy = 0.5)
        
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
    
        net.setInput(cv2.dnn.blobFromImage(frame, 2.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    
        out = net.forward()
        out = out[:, :19, :, :]
    
        assert(len(BODY_PARTS) == out.shape[1])
    
        points = []
        
        
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > thres else None)
            
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]
            
            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                
        t, _ = net.getPerfProfile()
        img = frame
        
        
        size = (frameWidth, frameHeight) 
        
        cv2.imshow('pose',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    cv2.destroyAllWindows()
    
        
    

output = pose_estimation(cap = cap)


cap.release()


cv2.waitKey(0)
cv2.destroyAllWindows()
    
    
            