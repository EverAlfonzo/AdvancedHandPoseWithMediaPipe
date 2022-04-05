import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import create_video

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def max_detection(file_name, vid_dir, padding=0, resize=False):
    cap = cv2.VideoCapture(os.path.join(vid_dir,f'{file_name}.mp4'))


    with mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("Ignoring empty camera: max_detection fase")
                break
            if resize:
                frame = cv2.resize(frame, resize)
            (h,w,c) = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    for lm in hand.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
    return (x_max+padding, y_max+padding, max(x_min-padding,0), max(y_min-padding,0))

def hands_extraction(file_name,vid_dir, resize=(299,299)):
    #file_name="WIFI"
    cap = cv2.VideoCapture(os.path.join(vid_dir,f'{file_name}.mp4'))

    fps = cap.get(cv2.CAP_PROP_FPS)
    output_image_array = []
    fourcc = 'DIVX'
    #TODO: detectar los maximos y minimos
    #x_max, y_max, x_min, y_min = max_detection(file_name, vid_dir, resize=resize)
    #print((x_max,y_max,x_min,y_min))

    #cap = cv2.VideoCapture(0)

    i=0
    os.mkdir(file_name)
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
        while cap.isOpened():
            i+=1
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            (h,w,c) = frame.shape
            frame = cv2.resize(frame, resize)
            #print((h,w,c))
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
            img = np.full((299,299,3), color, np.uint8)

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
            #print(f'shape before: {img.shape}')
            #img = img[y_max:y_min, x_max:x_min,:]
            
            #print(f'shape after: {img.shape}')
                cv2.imwrite(os.path.join(file_name, '{}.jpg'.format(str.rjust(str(i),6,'0'))), img)
            else:
                i-=1

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    create_video.create_video(img_dir=file_name,fps=fps, fourcc=fourcc, video=f"{file_name}.avi")

    #out.release()
    cap.release()
    cv2.destroyAllWindows()

def main():
    hands_extraction('marzo_kuki','videos_prueba')



if __name__ == '__main__':
    main()
