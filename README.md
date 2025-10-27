YOLOv8 Object Detection & Tracking Engine with Streamlit

 <!-- Optional: Add this badge AFTER deploying -->

<!-- Optional but HIGHLY recommended: Record a short GIF of your app working -->

Overview

This project implements a real-time object detection and tracking application using the YOLOv8 deep learning model and the DeepSORT algorithm. The interactive web interface is built with Streamlit, allowing users to process images, videos, or live webcam feeds.

Problem Statement

(Briefly describe the problem this project solves or the goal it achieves. E.g., "The goal was to create an accessible tool for real-time object identification and tracking applicable to various scenarios like security monitoring or activity analysis.")

Features

Object Detection: Utilizes the YOLOv8n model for fast and efficient object detection.

Object Tracking: Implements DeepSORT algorithm to assign and maintain unique IDs for detected objects across video frames.

Multiple Input Sources: Supports detection and tracking on uploaded images, uploaded videos, and live webcam feeds.

Real-time Performance: Displays calculated Frames Per Second (FPS) for video and webcam streams.

Interactive UI: Built with Streamlit, featuring adjustable confidence threshold and clear status indicators.

Robust Error Handling: Includes checks for model loading, file processing, and library compatibility.

Tech Stack

Model: YOLOv8n (from Ultralytics)

Tracking Algorithm: DeepSORT (deep-sort-realtime library)

Core Libraries: Python 3.10, PyTorch (CUDA enabled), OpenCV

Web Framework: Streamlit (streamlit, streamlit-webrtc)

Annotation/Utils: Supervision (supervision), NumPy

Environment: Conda

Setup and Installation

Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name


Create and activate Conda environment:

conda create -n yolo-env python=3.10
conda activate yolo-env


Install PyTorch with CUDA support (adjust cu121 if you used a different CUDA version):

pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)


Install dependencies:

python -m pip install -r requirements.txt


Usage

Run the Streamlit application from the project directory:

streamlit run app.py


Navigate through the tabs ("Image", "Video", "LIVE Webcam") to upload your media or start the webcam feed. Adjust the confidence threshold in the sidebar as needed.

Project Structure

project1-yolo/
├── app.py              # Main Streamlit application script
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── (Optional: demo.gif) # Demo GIF of the application


Potential Improvements

Allow selection of different YOLOv8 model sizes (s, m, l, x).

Add options for saving processed video output.

Implement custom model loading for user-trained weights.

Containerize the application using Docker for easier deployment.

Acknowledgements

Ultralytics YOLOv8

DeepSORT Realtime

Supervision

Streamlit

streamlit-webrtc