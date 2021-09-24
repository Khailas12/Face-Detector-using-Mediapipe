# from camera_position import camera_position
from face_det import FaceDetector
import concurrent.futures
import time
import cv2



class MainDet():
    
    def __enter__(self): 
        print('__enter__')
        return self
    
    
    def main(self):
        previous_time = 0
        current_time = 0
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        face_detector = FaceDetector()
        
        
        while cap.isOpened():
            success, camera = cap.read()
            camera = face_detector.find_face(camera)
            
            if not success:
                raise IOError('Ignoring the empty cam')
                continue
            
            self.land_mark_list = face_detector.camera_position(camera, draw=True)
            
            
            if len(self.land_mark_list) != 0:
                print(self.land_mark_list[4])
            
            current_time = time.time()
            fps = 1/(current_time - previous_time)
            previous_time = current_time
            
            
            cv2.putText(
                camera, str((fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 4
                )
        
            cv2.imshow('Face Detector', camera)
            
            if cv2.waitKey(1) & 0xFF==ord("q"):
                break


        cap.release()
        cv2.destroyAllWindows()
            
            
    def __exit__(self, type, value, traceback):
        return isinstance(value, TypeError)
    

                
if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        with MainDet() as fd:
            executor.map(fd.main())
            executor.map(fd.find_face())
            executor.map(fd.camera_position())