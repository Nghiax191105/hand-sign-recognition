# Hand Sign System
A real-time hand sign recognition project using OpenCV, MediaPipe Hands, and machine learnig.
## Overview
This project detects hand landmarks from camera input and classifies hand signs in real time.
## Pipeline
Camera  
→ OpenCV  
→ MediaPipe Hands  
→ Hand Landmark Features  
→ SVM / LSTM  
→ Predicted Sign
## Technologies
- Python
- OpenCV
- MediaPipe Hands
- NumPy
- scikit-learn
- SVM
- LSTM
## Main Features
- Real-time camera input
- Hand landmark extraction
- Static gesture data collection
- SVM model training
- Real-time gesture prediction
## Project Results
- 33 static gesture classes
- 120 samples per valid class
- 97.6% test accuracy on the project dataset
