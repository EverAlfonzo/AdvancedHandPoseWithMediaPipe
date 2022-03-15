import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture('scene2-camera1.mov')


with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
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
        img = np.full((860,720,3), color, np.uint8)
        # Detections
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                         )
            
        
        #cv2.imshow('Hand Tracking', image)
        cv2.imshow('Just the tracking',img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
