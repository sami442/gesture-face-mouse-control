# 🖱️ Gesture & Face Mouse Control

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.13-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

## 🎥 Live Demo
👉 [**Watch Demo Video**](https://www.loom.com/share/734671c6fdbf4e7ea253a3eee982ccb4)

## 📌 Overview
A real-time computer vision application that lets you control your 
mouse cursor entirely through hand gestures and face features — 
no physical mouse needed. Built using MediaPipe for landmark 
detection and PyAutoGUI for system-level mouse control.

This project demonstrates real-time computer vision, landmark 
detection, and human-computer interaction — a completely different 
technique from traditional image classification projects.

## 🎮 Gesture Controls

| Gesture | Action |
|---------|--------|
| ☝️ Move index finger | Move cursor |
| 👌 Pinch (index + thumb) | Left click |
| ✊ Fist | Right click |
| ☝️✌️ Index + Middle fingers together | Scroll UP |
| 💍🤙 Ring + Pinky fingers together | Scroll DOWN |
| 👍 Thumbs up | Open new Chrome tab |
| 😑 Close BOTH eyes for 2 seconds | Quit app |
| Q key | Quit app |

## ✨ Features
- 🖱️ Real-time cursor control via hand tracking
- 👆 Left and right click via hand gestures
- 📜 Scroll up and down with finger combinations
- 🌐 Open new browser tabs with thumbs up
- 😑 Graceful app exit with both eyes closed
- ⚡ Smooth cursor movement with configurable smoothing
- 🔒 Emergency stop — move mouse to top-left corner

## 🛠️ Tech Stack
| Technology | Purpose |
|------------|---------|
| Python 3.12 | Core language |
| MediaPipe 0.10.13 | Hand and face landmark detection |
| OpenCV | Webcam capture and frame processing |
| PyAutoGUI | System-level mouse and keyboard control |

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/sami442/gesture-face-mouse-control.git
cd gesture-face-mouse-control
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python main.py
```

### 4. Controls appear in terminal — start gesturing!

## ⚠️ Requirements
- Python 3.10+
- Webcam (built-in or external)
- Good lighting for reliable detection
- Windows OS (PyAutoGUI system control)

## 💡 Tips for Best Results
| NumPy | Numerical operations and distance calculations |

## ⚙️ How It Works
