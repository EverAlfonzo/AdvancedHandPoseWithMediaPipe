import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# # For static images:
# IMAGE_FILES = []
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#   for idx, file in enumerate(IMAGE_FILES):
#     # Read an image, flip it around y-axis for correct handedness output (see
#     # above).
#     image = cv2.flip(cv2.imread(file), 1)
#     # Convert the BGR image to RGB before processing.
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Print handedness and draw hand landmarks on the image.
#     print('Handedness:', results.multi_handedness)
#     if not results.multi_hand_landmarks:
#       continue
#     image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       print('hand_landmarks:', hand_landmarks)
#       print(
#           f'Index finger tip coordinates: (',
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#       )
#       mp_drawing.draw_landmarks(
#           annotated_image,
#           hand_landmarks,
#           mp_hands.HAND_CONNECTIONS,
#           mp_drawing_styles.get_default_hand_landmarks_style(),
#           mp_drawing_styles.get_default_hand_connections_style())
#     cv2.imwrite(
#         '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
#     # Draw hand world landmarks.
#     if not results.multi_hand_world_landmarks:
#       continue
#     for hand_world_landmarks in results.multi_hand_world_landmarks:
#       mp_drawing.plot_landmarks(
#         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)


def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
            
            output = text, coords
            
    return output

# For webcam input:
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    h, w, c = image.shape
    print(f"With: {w}, height: {h}, chanels: {c}")
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    #Black Background
    color = (0,0,0)
    blk = np.full((h,w,c), color, np.uint8)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hands_crop = []
    hands_side = []
    if results.multi_hand_landmarks:
      #import pdb; pdb.set_trace()
      for num, hand in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(
            blk,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))

        

      #for num, hand in enumerate(results.multi_hand_landmarks):
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        padding= 20
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
        #cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        #cv2.circle(image,(x_min,y_min), 2,(0,0,255), 2)
        #cv2.circle(image,(x_max,y_max), 2,(0,255,0), 2)
        #mp_drawing.draw_landmarks(blk, hand, mp_hands.HAND_CONNECTIONS)
        hands_crop.append(blk[y_min-padding:y_max+padding, x_min-padding:x_max+padding,:])
        if get_label(num, hand, results):
            text, coord = get_label(num, hand, results)
            hands_side.append(text)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    print(len(hands_side))
    print(len(hands_crop))
    if len(hands_crop)>0 and hands_crop[0].any():
        if len(hands_side)>0 and 'left' in hands_side[0]:
          cv2.imshow(f'Hands Izquierda',hands_crop[0])
        elif len(hands_side)>0 and 'left' not in hands_side[0] and len(hands_crop)>1 and hands_crop[1].any():
          cv2.imshow(f'Hands Izquierda', hands_crop[1])
        else:
          cv2.imshow(f'Hands Izquierda', hands_crop[0])


    if len(hands_crop)>1 and hands_crop[1].any():
        if len(hands_side)>1 and 'right' in hands_side[1]:
            cv2.imshow(f'Hands Derecha',hands_crop[1])
        elif len(hands_side)>1 and 'right' not in hands_side[1]:
            cv2.imshow(f'Hands Derecha', hands_crop[0])
        else:
            cv2.imshow(f'Hands Derecha',hands_crop[1])

    #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), blk)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()