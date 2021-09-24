import cv2
from face_detector import FaceDetector



def camera_position(camera, face_num=0, draw=True):
    land_mark_list = []
    
    detect = FaceDetector()
    results = detect.process(camera)
    
    if results.detections:
        my_camera = results.detections[face_num]
        
        for id, lm in enumerate(my_camera.landmark):
            height, width, channels = camera.shape
            
            cx, cy = int(lm.x*width), int(lm.y*height)
            land_mark_list.append([id, cx, cy])
            
        if draw:
            cv2.circle(camera, (cx, cy), 15, (255, 0, 255),cv2.FILLED)
            
    return land_mark_list   