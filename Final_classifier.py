import pickle
import cv2
import mediapipe as mp
import numpy as np
import accuracy_score

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j",
              10: "k", 11: "l", 12: "m", 13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t",20: "u", 21: "v", 
              22: "w", 23: "x", 24: "y", 25: "z", 26: "0", 27: "1", 28: "2", 29: "3", 30: "4", 31: "5", 32: "6", 33: "7", 34: "8", 35: "9",}

# Load testing data
data_dict = pickle.load(open('./data.pickle', 'rb'))
x_test = np.asarray(data_dict['data'])
y_test = np.asarray(data_dict['labels'])

predicted_labels = []
correct_label = None
total_frames = 0
correct_predictions = 0

while True:
    data_aux = []
    x_ = []
    y_ = []
    
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_label = int(prediction[0])
        except:ValueError
        
        # Update the correct label for the captured frame
        for key in labels_dict:
            correct_label = predicted_label# Replace with the correct label index (0 for 'A', 1 for 'B', etc.)
        
        if predicted_label == correct_label:
            correct_predictions += 1
        
        total_frames += 1

        # Calculate test accuracy
        test_accuracy = accuracy_score.accuracy_score()

        # Overlay test accuracy and predicted result on the frame
        cv2.putText(frame, f"Predicted: {labels_dict[predicted_label]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Test Accuracy: {test_accuracy:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, labels_dict[predicted_label], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(10)
cv2.destroyAllWindows()









