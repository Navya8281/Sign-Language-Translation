# Sign-Language-Translation

Overview of Steps
   1. Install and import dependencies.
   2.  Extract key points using MediaPipe Holistic.
   3.  Collect key points data into NumPy arrays.
   4.  Pre-process data and prepare training dataset.
   5.  Build and train LSTM neural network.
   6.  Evaluate the model and visualize predictions in real-time.
   7. User Interface Creation
Dependencies to Install
Libraries to install:
    tensorflow
    opencv-python
    mediapipe
    sklearn
    matplotlib
Importing Libraries
    OpenCV: For webcam access.
    NumPy: For data manipulation.
    Matplotlib: For data visualization.
    MediaPipe: For keypoint extraction.
Data Collection and Processing Steps
    1. Webcam Access
        Access webcam using OpenCV.
        Loop through frames, capturing data.
    2. MediaPipe Initialization
        Create variables for MediaPipe Holistic and drawing utilities.
        Functions for detection and drawing landmarks.
    3. Collect Data for LSTM
        Collect key points from hands, body, and face in NumPy arrays.
        Organize data by actions (e.g., "hello", "thanks", "I love you").
        Each action will have 30 sequences of 30 frames each—total of 90x30x1662 data points.
    4. Pre-process Data
        Create sequences and labels.
        Split data into training and testing sets.
    5. Building the LSTM Model
        Model architecture:
            Sequential model with LSTM layers.
            Three LSTM layers followed by dense layers.
            Final layer uses softmax activation to predict classes.
        Model Compilation
            Compile model using:
            Optimizer: Adam
            Loss: Categorical Crossentropy
            Metrics: Categorical Accuracy
        Model Training
            Train the model with the training dataset.
            Option to monitor performance using TensorBoard.
    6. Evaluating the Model's Performance
        Use confusion matrix and accuracy score to evaluate model.
        Predictions accuracy is assessed on test data.
        Real-time Testing and Visualization
    7.  User Interface using Streamlit
        UI without model confidence score - trial.py
        UI with model confidence score - demo.py    
        