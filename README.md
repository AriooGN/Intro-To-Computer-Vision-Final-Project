# Intro To Computer Vision Final Project

Welcome to the Introduction to Computer Vision Final Project repository. This project implements **real-time facial recognition and detection** using OpenCV and DeepFace.

## 📋 Project Overview

This repository contains a comprehensive computer vision application that performs **real-time face detection** and **identity recognition** from webcam input. The project combines OpenCV's DNN-based face detector with DeepFace's deep learning models to identify and track individuals in video streams.

## 🎯 Objectives

- Implement real-time face detection using OpenCV's SSD ResNet model
- Perform facial identity recognition using multiple deep learning backends (VGG-Face, FaceNet, ArcFace)
- Create an interactive interface for webcam-based facial recognition
- Enable dynamic registration of new faces from live video
- Optimize performance through frame skipping and async processing
- Demonstrate practical computer vision techniques in a production-ready application

## 🛠️ Technologies & Tools

- **Python 3.7+** - Primary programming language
- **OpenCV (cv2)** - Face detection using DNN module with pre-trained SSD ResNet
- **DeepFace** - Facial recognition and identity verification (supports VGG-Face, FaceNet, ArcFace models)
- **TensorFlow/Keras** - Backend for deep learning models
- **NumPy** - Numerical computing and array operations
- **Threading** - Background processing for smooth video performance

## 📁 Project Structure

```
Intro-To-Computer-Vision-Final-Project/
├── README.md                      # This file
├── facial-recognition/            # Main application directory
│   ├── main.py                   # Entry point with video capture loop and UI
│   ├── detector.py               # OpenCV DNN face detection module
│   ├── recognizer.py             # DeepFace-based facial recognition with threading
│   ├── utils.py                  # Helper functions (FPS, cropping, drawing, etc.)
│   ├── config.py                 # Configuration and thresholds
│   ├── requirements.txt           # Python dependencies
│   ├── known_faces/              # Directory for storing reference images
│   │   └── person_name/          # Sub-directory per person
│   │       ├── img1.jpg
│   │       └── img2.jpg
│   └── models/                   # Downloaded detection model weights
│       ├── deploy.prototxt
│       └── res10_300x300_ssd_iter_140000.caffemodel
└── .gitattributes                # Git configuration
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Webcam connected to your computer

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AriooGN/Intro-To-Computer-Vision-Final-Project.git
cd Intro-To-Computer-Vision-Final-Project
```

2. Navigate to the facial-recognition directory:
```bash
cd facial-recognition
```

3. Create a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

The application will automatically download the OpenCV DNN model weights on first run.

## 📖 Usage

### Basic Usage

Run the main application:
```bash
python main.py
```

### Command Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--camera` | `0` | Webcam index (0 for default) |
| `--model` | `VGG-Face` | Recognition backend: `VGG-Face`, `Facenet`, or `ArcFace` |
| `--det-conf` | `0.5` | Minimum detection confidence (0.0-1.0) |
| `--recognition-interval` | `5` | Run recognition every N frames |
| `--min-face` | `40` | Minimum face size in pixels (at full resolution) |
| `--detection-scale` | `0.5` | Resize factor for detection (0.5 = 50% speed improvement) |

**Example:**
```bash
python main.py --model Facenet --det-conf 0.6 --camera 0
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `m` | Cycle recognition model (VGG-Face → Facenet → ArcFace) |
| `s` | Save screenshot to project root |
| `a` | Register a new face (prompts for name, saves largest detected face) |

### Setting Up Known Faces

Create a `known_faces` directory structure:
```
known_faces/
  John_Doe/
    img1.jpg
    img2.jpg
    img3.jpg
  Jane_Smith/
    img1.jpg
    img2.jpg
```

**Recommended:** Use 1-5 reference images per person for optimal recognition performance. The application supports `.jpg`, `.png`, and other standard image formats.

**Alternative (Dynamic Registration):**
Press `a` while a face is detected to capture and register it interactively.

## 🔍 Key Features

- **Real-Time Face Detection** - OpenCV DNN SSD ResNet detects faces on every frame with configurable confidence thresholds
- **Identity Recognition** - DeepFace identifies individuals against a local gallery of known faces
- **Multi-Model Support** - Choose between VGG-Face, FaceNet, and ArcFace with different accuracy/speed trade-offs
- **Async Processing** - Recognition runs in a background thread to keep video preview smooth and responsive
- **Dynamic Face Registration** - Add new people to the gallery interactively from live video
- **Performance Optimization** - Frame downscaling for detection and frame skipping for recognition reduce computational load
- **Visual Feedback** - Real-time display of detection confidence, recognition accuracy, and FPS counter
- **Configurable Thresholds** - Model-specific distance thresholds aligned with DeepFace verification metrics

## 🏗️ Architecture

### Core Modules

- **`main.py`** — Main video capture loop, frame processing pipeline, HUD rendering, and keyboard event handling
- **`detector.py`** — OpenCV DNN face detector wrapper with automatic model weight downloading
- **`recognizer.py`** — Background thread worker for DeepFace identity matching with queue-based job handling
- **`utils.py`** — Utility functions for FPS calculation, box scaling, face cropping, and rendering overlays
- **`config.py`** — Centralized configuration for thresholds, model paths, and hyperparameters

### Processing Pipeline

1. **Frame Capture** - Read frame from webcam at native resolution
2. **Detection (Every Frame)** - Optionally downscale, detect faces using OpenCV DNN, scale boxes back
3. **Recognition (Every N Frames)** - Crop faces, submit async recognition jobs to background worker
4. **Results Display** - Overlay detection boxes and recognition labels on live video
5. **Keyboard Input** - Handle user interactions for model cycling, face registration, and screenshots

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Recognition distance thresholds (per model)
RECOGNITION_THRESHOLDS = {
    "VGG-Face": 0.40,
    "Facenet": 0.40,
    "ArcFace": 0.68,
}

# Detection confidence minimum
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# Frame skipping for recognition
RECOGNITION_FRAME_INTERVAL = 5

# Minimum face size to process
MIN_FACE_SIZE = 40

# Detection downscaling factor
DETECTION_INPUT_SCALE = 0.5
```

## 📊 Performance Notes

- **GPU Acceleration** - TensorFlow/DeepFace will automatically use GPU if CUDA is installed and configured
- **Empty Gallery** - All detected faces show as "Unknown" until reference images are added
- **FPS Optimization** - Reduce FPS impact by:
  - Lowering `--detection-scale` (e.g., 0.3 for faster detection)
  - Increasing `--recognition-interval` (e.g., 10 or higher)
  - Using lighter models (Facenet is faster than VGG-Face)
  - Enabling GPU acceleration

## 📚 References

- [OpenCV Documentation](https://docs.opencv.org/)
- [OpenCV DNN Face Detector](https://docs.opencv.org/4.x/d4/d12/classcv_1_1dnn_1_1Net.html)
- [DeepFace GitHub](https://github.com/serengp/deepface)
- [Face Recognition with Python](https://github.com/ageitgey/face_recognition)

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements, bug fixes, or feature enhancements.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✉️ Contact

For questions or feedback about this project, please open an issue in the GitHub repository.

---

**Last Updated:** May 1, 2026  
**Author:** Arian (AriooGN)  
**Project Type:** Computer Vision - Facial Recognition & Detection
