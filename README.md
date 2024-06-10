# PRODIGY_ML_04

## Hand Gesture Recognition Model

### Task Overview
This task involves developing a hand gesture recognition model that can accurately identify and classify different hand gestures from image or video data. The goal is to enable intuitive human-computer interaction and gesture-based control systems.

### Dependencies
The task requires the following libraries:

os
cv2
numpy
scikit-learn
tensorflow
matplotlib

### How the Code Works
#### 1. Load and Preprocess Images
Images are loaded from a specified dataset directory. Each image is converted to grayscale, resized to 64x64 pixels, and sharpened using a predefined kernel. The labels are mapped from the folder names corresponding to each gesture.

#### 2. Normalize Images
The pixel values of the images are normalized by scaling them to the range [0, 1].

#### 3. Reshape Images
Images are reshaped to fit the input shape required by the CNN model (64x64x1).

#### 4. One-hot Encode Labels
The labels are one-hot encoded to facilitate training the model.

#### 5. Train-Test Split
The dataset is split into training and testing sets with an 80-20 split.

#### 6. Build CNN Model
A Convolutional Neural Network (CNN) is built using the following layers:

##### Convolutional layers with ReLU activation and MaxPooling
##### Flatten layer to convert 2D data to 1D
##### Dense layers with ReLU activation and Dropout for regularization
##### Output layer with softmax activation for multi-class classification

#### 7. Compile and Train the Model
The model is compiled with Adam optimizer and categorical cross-entropy loss. Data augmentation is applied using ImageDataGenerator to improve model generalization. The model is trained for 20 epochs with a batch size of 32.

#### 8. Evaluate the Model
The model's performance is evaluated on the test set, providing accuracy and loss metrics. Training and validation accuracy/loss are plotted for visualization.

#### 9. Predict and Visualize Results
The trained model is used to predict labels for the test set. A few sample predictions are visualized along with their true labels to assess the model's performance.

### Conclusion
This project demonstrates the development of a hand gesture recognition model using a Convolutional Neural Network (CNN). The model effectively classifies different hand gestures, enabling intuitive human-computer interaction and gesture-based control systems.
