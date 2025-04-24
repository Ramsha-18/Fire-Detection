

---

# 🔥 Fire-Detection

**Fire-Detection** is an AI-powered solution for real-time fire detection using advanced computer vision techniques. Built on **YOLOv8 (You Only Look Once)**, this system provides fast and accurate detection of fire in diverse environments. It is designed for early alerting to enhance safety and prevent potential fire disasters.

---

## 🚀 Features

- **Real-time Detection** – High-speed inference using YOLOv8.
- **Customizable Models** – Easily trainable for different fire scenarios.
- **Easy Integration** – Compatible with various systems for seamless deployment.
- **Alert System** – Configurable alerts to enable prompt response.
- **Open Source** – Community contributions are welcome to improve the system.

---

## 🛠️ Installation

Follow these steps to set up the project:

### 1. Clone the Repository

```bash
git clone https://github.com/Ramsha-18/ultralytics.git
cd ultralytics/Fire-Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download YOLOv8 Model Weights

```bash
wget https://github.com/ultralytics/yolov8/releases/download/v8.0/yolov8n.pt
```

---

## 🎬 Usage

### 🔹 Command Line Interface (CLI)

Run YOLOv8 on an input video:

```bash
yolo predict model=yolov8n.pt source='path_to_video.mp4'
```

### 🔹 Python Script

```python
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video source
cap = cv2.VideoCapture('path_to_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Display results
    annotated_frame = results[0].plot()
    cv2.imshow('Fire Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🧠 Training on a Custom Dataset

1. **Prepare Your Dataset**
   - Annotate and format your dataset in YOLO format.

2. **Edit Config Files**
   - Modify the model config (`.yaml`) and dataset config (`your_dataset.yaml`) accordingly.

3. **Train the Model**

```bash
yolo train model=yolov8n.yaml data=your_dataset.yaml epochs=50
```

---

## 🤝 Contributing

We welcome community contributions!  
Feel free to:
- Submit pull requests
- Open issues for bugs or suggestions
- Share improvements and features

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🙏 Acknowledgments

Thanks to the amazing team at [Ultralytics](https://github.com/ultralytics) for their efforts in building and maintaining the YOLO series.

---
