import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3,max_num_hands = 5)

labels_dict = {0: 'A', 1: 'B', 2: 'L'}
while cv2.waitKey(1) != 27:

    data_aux = []
    x_ = []
    y_ = []



    ret, frame = cap.read()#read the frame
    
    
    H, W, _ = frame.shape #get the height and the width of the frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert the frame to RGB encoding

    results = hands.process(frame_rgb) #using hands mediapipe utility, detect hands in the frame

    
    if results.multi_hand_landmarks:
        print("Length ",len(results.multi_hand_landmarks))
        #print(type(results.multi_hand_landmarks))
        
        #draw hand landmarks for each hand
        for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


        hand_dict = {}

        #create a dictionary that comprehends all of the detected hands, their correspective landmarks, and 3 points(x,y,z)
        for hand_index,hand in enumerate(results.multi_hand_landmarks):
            hand_dict["hand"+str(hand_index)] = {("landmark"+str(hand_landmark_index)):{"x":landmark.x, "y":landmark.y, "z": landmark.z} for hand_landmark_index, landmark in enumerate(hand.landmark) }


        for i,h_index in enumerate(hand_dict):
            landmarks_dict = hand_dict[h_index]

            data_aux.append([])
            x_.append([])
            y_.append([])

            for land_index in landmarks_dict:
                x = landmarks_dict[land_index]["x"]
                y = landmarks_dict[land_index]["y"]

                x_[i].append(x)
                y_[i].append(y)
            
            for land_index in landmarks_dict:
                x = landmarks_dict[land_index]["x"]
                y = landmarks_dict[land_index]["y"]

                data_aux[i].append(x - min(x_[i]))
                data_aux[i].append(y - min(y_[i]))
            
            x1 = int(min(x_[i]) * W) - 10
            y1 = int(min(y_[i]) * H) - 10
            x2 = int(max(x_[i]) * W) - 10
            y2 = int(max(y_[i]) * H) - 10

            prediction = model.predict([np.asarray(data_aux[i])])



            predicted_character = labels_dict[int(prediction[0])]
            print(h_index, prediction, predicted_character)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                       cv2.LINE_AA)
    
        """
        
        for hand_index,hand in enumerate(results.multi_hand_landmarks):
            
            #print(hand.landmark[0].x)
            #hand.landmark is a list of landmarks each of which consists of three points, x, y, z
            print(len(hand.landmark))
            
            for hand_landmark in hand.landmark:
                #print(hand_landmark)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10


           
            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            #cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
            #           cv2.LINE_AA)
            """

    cv2.imshow('frame', frame)
    


cap.release()
cv2.destroyAllWindows()
