Emotion Detection Using CNN
This project implements a real-time emotion detection system using a Convolutional Neural Network (CNN). The model is trained on facial expression data and can predict emotions from live video or images.

Project Structure :-
📂 Emotion-Detection  
├── 📜 cnn.ipynb              # Jupyter notebook for CNN model development  
├── 📜 emotiondetector.h5     # Trained CNN model weights  
├── 📜 emotiondetector.json   # Model architecture in JSON format  
├── 📜 README.md              # Project documentation  
├── 📜 realtimedetection.py   # Python script for real-time emotion detection  
├── 📜 requirements.txt       # List of required dependencies  
└── 📜 trainmodel.ipynb       # Jupyter notebook for training the CNN model  
Setup and Installation
Clone the repository :-
    git clone https://github.com/animesh33-ctrl/Facial-Emotion-Recognition
    cd Emotion-Detection

Install dependencies :-
    pip install -r requirements.txt

Run the real-time detection script :-
    python realtimedetection.py

Model Training
    To train the CNN model, open trainmodel.ipynb and execute the cells step by step. The trained model is saved as emotiondetector.h5 and its architecture as emotiondetector.json.

Real-Time Emotion Detection
    Run realtimedetection.py to capture live video from the webcam and classify emotions in real time.

Dependencies :-
    Python 3.x
    TensorFlow/Keras
    OpenCV
    NumPy
    Matplotlib
Install all required packages using :-
    pip install -r requirements.txt

License :-
    This project is open-source under the MIT License.

