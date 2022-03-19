import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import create_video

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture('scene2-camera1.mov')

fps = cap.get(cv2.CAP_PROP_FPS)
output_image_array = []
fourcc = 'DIVX'
size=(860,720)

#cap = cv2.VideoCapture(0)

i=0
os.mkdir('semana_papa')
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        i+=1
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        ret, frame = cap.read()
        (h,w,c) = frame.shape
        #if i==1:
        #    out = cv2.VideoWriter('project.avi',fourcc, fps, (h,w))

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # FONDO NEGRO
        color = (0,0,0)
        # IMAGEN DE 860x720 x3 canales
        img = np.full((h,w,3), color, np.uint8)
        # Detections
        
        # Rendering results
        if results.multi_hand_landmarks:
            
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                                         )
        #out.write(img)
        #cv2.imshow('Hand Tracking', image)
        #cv2.imshow('Just the tracking',img)
        
        cv2.imwrite(os.path.join('semana_papa', '{}.jpg'.format(str.rjust(str(i),6,'0'))), img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

create_video.create_video(img_dir='semana_papa',fps=5, fourcc=fourcc, video="project.avi")

#out.release()
cap.release()
cv2.destroyAllWindows()
