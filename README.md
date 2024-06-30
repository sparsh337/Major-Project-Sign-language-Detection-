# Major-Project-Sign-language-Detection-

#1 Collect_image Code.
The code initializes a webcam to capture and save images for multiple classes. It creates directories for each class and captures a specified number of images upon user confirmation. The images are saved in their respective class directories. Error handling is included to ensure the webcam is correctly accessed and frames are captured without issues. Finally, the webcam is released, and OpenCV windows are closed.

#2 Create Dataset Code.
The code processes hand gesture images to create a dataset for sign language detection. It initializes MediaPipe Hands for landmark detection, reads images from specified directories, and extracts and normalizes hand landmarks. The landmarks and their corresponding labels are stored in lists and saved as a pickle file for future use in machine learning models.

#3 Train Classifier code.
The code loads preprocessed hand gesture data, splits it into training and test sets, and trains various classifiers to detect sign language gestures. It evaluates the model's performance by predicting the test set labels and calculating the accuracy score. The code loads preprocessed hand gesture data, splits it into training and test sets, and trains various classifiers, including a Convolutional Neural Network (CNN), to detect sign language gestures. It evaluates the models' performance using accuracy scores and saves the trained models to pickle files for later use.

#4 Final Classifier code.
The code captures real-time video from the webcam, detects hand landmarks using MediaPipe, and uses a pre-trained model to classify hand gestures into sign language symbols. It displays the predicted gesture and accuracy on the video feed.
