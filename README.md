
# Person Detection and Tracking Using YOLOv8 and DeepSORT

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the YOLOv8 Model](#training-the-yolov8-model)
- [Running Inference and Tracking](#running-inference-and-tracking)
- [Analyzing Model Predictions](#analyzing-model-predictions)
- [Reproducing the Results](#reproducing-the-results)
- [Source Code Files](#source-code-files)
- [Test Video Outputs](#test-video-outputs)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

This project focuses on detecting and tracking people in videos using a YOLOv8 model combined with the DeepSORT tracking algorithm. The YOLOv8 model is trained to detect persons, and DeepSORT is used to assign unique IDs to each individual, allowing for consistent tracking even if they temporarily exit and re-enter the frame.

## Project Structure

```
├── dataset_preparation.py     # Script to prepare the dataset
├── train_yolov8.py            # Script to train the YOLOv8 model
├── inference_tracking.py      # Script to run inference and tracking
├── requirements.txt           # Required Python packages
├── data.yaml                  # YOLO data configuration file
├── output_with_detections.mp4 # Output video with detections (if included)
├── README.md                  # Project documentation
└── ...                        # Other files and directories
```

## Prerequisites

- Python 3.7 or higher
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [DeepSORT-Realtime](https://pypi.org/project/deep-sort-realtime/)
- OpenCV
- PyTorch

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages**

   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a `requirements.txt` file, you can install the packages individually:

   ```bash
   pip install -U ultralytics
   pip install deep-sort-realtime opencv-python torch
   ```

## Dataset Preparation

### 1. Download the Dataset

- Download the public dataset from Kaggle used for training. Ensure that the dataset contains images and corresponding label files in YOLO format.

### 2. Select a Subset of Images

Use the `dataset_preparation.py` script to select 4000 random images and their corresponding labels from the training dataset.

**dataset_preparation.py**

```python
import os
import random
import shutil

# Paths to your directories
train_images_dir = r"C:\path\to\dataset\train\images"
train_labels_dir = r"C:\path\to\dataset\train\labels"
output_images_dir = r"C:\path\to\dataset\new_train\images"
output_labels_dir = r"C:\path\to\dataset\new_train\labels"

# Ensure the output directories exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Get all image file names
image_files = os.listdir(train_images_dir)

# Randomly select 4000 images
selected_images = random.sample(image_files, 4000)

# Copy the selected images and corresponding labels to the output directory
for image_file in selected_images:
    # Copy image
    src_image_path = os.path.join(train_images_dir, image_file)
    dst_image_path = os.path.join(output_images_dir, image_file)
    shutil.copy(src_image_path, dst_image_path)
    
    # Copy corresponding label
    label_file = os.path.splitext(image_file)[0] + '.txt'  # handles different image extensions
    src_label_path = os.path.join(train_labels_dir, label_file)
    dst_label_path = os.path.join(output_labels_dir, label_file)
    shutil.copy(src_label_path, dst_label_path)

print("Successfully copied 4000 images and their corresponding labels.")
```

- **Instructions:**
  - Replace the paths with the actual paths to your dataset directories.
  - Run the script to copy the selected images and labels.

### 3. Update the Data Configuration File

Create or update `data.yaml` with the following content:

```yaml
train: C:\path\to\dataset\new_train\images
val: C:\path\to\dataset\val\images  # Update with your validation images path

nc: 1  # Number of classes
names: ['person']  # Class names
```

- Ensure that the paths and class names are correctly specified.

## Training the YOLOv8 Model

Use the `train_yolov8.py` script to train the YOLOv8 model.

**train_yolov8.py**

```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')  # Load the pre-trained YOLOv8m model
model.train(data=r"C:\path\to\data.yaml",
            epochs=10,
            imgsz=640,
            batch=16,
            name='yolov8_person_detection',
            device='cpu',
            verbose=True)
```

- **Instructions:**
  - Replace `C:\path\to\data.yaml` with the path to your `data.yaml` file.
  - Adjust parameters like `epochs`, `imgsz`, `batch`, and `device` as needed.
  - Run the script to start training.

## Running Inference and Tracking

Use the `inference_tracking.py` script to perform inference and tracking on a video.

**inference_tracking.py**

```python
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the trained YOLOv8 model
model = YOLO(r"C:\path\to\your\trained_model.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=1800, nn_budget=100)

# Process video
cap = cv2.VideoCapture(r"C:\path\to\input_video.mp4")

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_detections.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Extract detections
    detections = []
    for result in results:
        for bbox, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if conf > 0.5 and int(cls) == 0:  # Only 'person' class
                x1, y1, x2, y2 = bbox.tolist()
                detections.append(([x1, y1, x2, y2], conf.item(), cls.item()))

    # Update tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

cap.release()
out.release()
```

- **Instructions:**
  - Replace the paths with the actual paths to your trained model and input video.
  - Run the script to generate `output_with_detections.mp4`.

## Analyzing Model Predictions

### Logic Behind Analyzing Predictions:

1. **Detection Filtering:**
   - The model predicts bounding boxes, class labels, and confidence scores.
   - Only detections with a confidence score above 0.5 and class ID 0 ('person') are considered.

2. **Tracking with DeepSORT:**
   - The filtered detections are passed to the DeepSORT tracker.
   - DeepSORT assigns a unique ID to each person and tracks them across frames.
   - It maintains the identity of individuals even if they temporarily leave and re-enter the frame.

3. **Visualization:**
   - Bounding boxes are drawn around each detected person.
   - Unique IDs are displayed above the bounding boxes.
   - The output video shows the tracking of individuals throughout the video.

### Key Parameters:

- **`max_age=1800`:** Allows the tracker to keep a person's ID for 1800 frames even
