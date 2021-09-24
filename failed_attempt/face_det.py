import mediapipe as mp
import concurrent.futures
import cv2
import time



class FaceDetector():

    def __init__(
        self, static_image_mode = False, min_detection_confidence=0.5, min_tracking_confidence=0.):
        
        self.static_image_mode = static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_face_detector = None
        self.drawing = None
        
      
        self.mp_face_detector = mp.solutions.face_detection
        
        self.detector = self.mp_face_detector.FaceDetection(
            self.static_image_mode, self.min_detection_confidence, self.min_tracking_confidence 
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        
    def find_face(self, camera, draw=True):    
        camera_rgb = cv2.cvtColor(cv2.flip(camera, 1), cv2.COLOR_BGR2RGB)
        
        camera.flags.writeable = False
        self.results = self.detector.process(camera)
        
        camera.flags.writeable = True
        camera = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR)
        
        if self.results.detections:
            for detection in self.results.detections:
                if draw:
                    self.mp_drawing.mp_drawing(camera, self.mp_face_detector, detection)
                        
        return camera
    
    
    def camera_position(self, camera, face_num=0, draw=True):
        land_mark_list = []
    
        if self.results.detections:
            my_camera = self.results.detections[face_num]
            
            for id, lm in enumerate(my_camera.landmark):
                height, width, channels = camera.shape
                
                cx, cy = int(lm.x*width), int(lm.y*height)
                land_mark_list.append([id, cx, cy])
                
            if draw:
                cv2.circle(camera, (cx, cy), 15, (255, 0, 255),cv2.FILLED)
                
        return land_mark_list   