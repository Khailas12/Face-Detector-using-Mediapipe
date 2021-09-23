import mediapipe as mp
import cv2


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


image_files = []

with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:

    for idx, file in enumerate(image_files):
        image = cv2.imread(file)
        
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))    # converting BGR to RGB
        
        
        if not results.detections:
            continue
        
        annotated_image = image.copy()                                 
        for detection in results.detections:
            print('Nose tip')

        print(mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.NOSE_TIP
            ))
        
        mp_drawing.draw_detection(annotated_image, detection)
        cv2.imwrite('tmp/annotated_image' + str(idx) + '.png', annotated_image)
        
        

# webcam input
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection:
    
    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
            print('Ignoring empty camera frame')
            continue
        
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                
        cv2.imshow('Face Detector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release() 