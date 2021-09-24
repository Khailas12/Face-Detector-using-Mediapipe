import mediapipe as mp
import cv2


class FaceDetector():

    def __init__(
        self, static_image_mode = False, max_faces=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.):
        
        
        self.static_image_mode = static_image_mode
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_face_detector = None
        self.drawing = None
        
      
        self.mp_face_detector = mp.solutions.face_detection
        
        self.detector = self.mp_face_detector.FaceDetection(
            self.static_image_mode, 
            self.max_faces,
            self.min_detection_confidence, 
            self.min_tracking_confidence 
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawSpec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
        
        
    def find_face(self, camera, draw=True):    
        self.camera_rgb = cv2.cvtColor(cv2.flip(camera, 1), cv2.COLOR_BGR2RGB)
        
        camera.flags.writeable = False
        self.results = self.detector.process(self.camera_rgb)
        
        camera.flags.writeable = True
        camera = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR)
        
        
        if self.results.multi_face_landmarks:
            for detection in self.results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        camera, self.mp_face_detector.FACE_CONNECTIONS,
                        detection, self.drawSpec, self.drawSpec
                        )
                face = []
                        
        return camera
    
    
    def camera_position(self, camera, face_num=0, draw=True):
        land_mark_list = []
    
        if self.results.multi_face_landmarks:
            my_camera = self.results.multi_face_landmarks[face_num]
            
            for id, lm in enumerate(my_camera.landmark):
                height, width, channels = camera.shape
                
                cx, cy = int(lm.x*width), int(lm.y*height)
                land_mark_list.append([id, cx, cy])
                
            if draw:
                cv2.circle(camera, (cx, cy), 15, (255, 0, 255),cv2.FILLED)
                
        return land_mark_list   