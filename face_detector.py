import mediapipe as mp
import concurrent.futures
import cv2
import time



class FaceDetector:
    def __init__(self):
        self.mp_face_detector = None
        self.drawing = None

    def __enter__(self): 
        print('__enter__')
        return self

    def my_face_detection(self):
        
        self.mp_face_detector = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        previous_time = 0
        current_time = 0
        
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

        
        with self.mp_face_detector.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
            ) as face_detection:
            
            while cap.isOpened():
                success, camera = cap.read()
                
                if not success:
                    raise IOError('ignoring empty camera frame')
                    continue
                
                camera = cv2.cvtColor(cv2.flip(camera, 1), cv2.COLOR_BGR2RGB)
                
                camera.flags.writeable = False
                self.results = face_detection.process(camera)
                
                
                camera.flags.writeable = True
                camera = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR)
                
                if self.results.detections:
                    for detection in self.results.detections:
                        self.mp_drawing.draw_detection(camera, detection)
                        
                    cv2.imshow('face detector', camera)
                    
                    if cv2.waitKey(1) & 0xFF==ord("q"):
                        break
                    
        cap.release()
        cv2.destroyAllWindows() 
        
        
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
    
    def __exit__(self, type, value, traceback):
        return isinstance(value, TypeError)
    

                
if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        with FaceDetector() as fd:
            executor.map(fd.my_face_detection())
            executor.map(fd.camera_position())
