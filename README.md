
# Person Detection and Tracking Using YOLOv8 and DeepSORT

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the YOLOv8 Model](#training-the-yolov8-model)
- [Running Inference and Tracking](#running-inference-and-tracking)
- [Analyzing Model Predictions](#analyzing-model-predictions)
- [Improvements](#Imporvements)
- [Test Video Outputs](#test-video-outputs)
---

## Introduction

This project focuses on detecting and tracking people in videos using a YOLOv8 model combined with the DeepSORT tracking algorithm. The YOLOv8 model is trained to detect persons, and DeepSORT is used to assign unique IDs to each individual, allowing for consistent tracking even if they temporarily exit and re-enter the frame.





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


2. **Install the required packages**

   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a `requirements.txt` file, you can install the packages individually:

   ```bash
   pip install -U ultralytics
   pip install deep-sort-realtime opencv-python torch
   ```

## Dataset Preparation



### 1. Creating a Dataset from the provided videos

The dataset used for this project is made up of videos provided as part of the assignment. These videos were processed by extracting each frame and treating it as an individual image. We used a pre-trained YOLOv8 model to detect and label only the "person" class in these images. Labels are generated in the YOLO format, and any images that do not contain a person are automatically removed from the dataset to optimize storage and performance.

The following script processes the images and generates the required labels:


```python
from ultralytics import YOLO
import os
import cv2

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Using YOLOv8 Nano for fast processing

# Directory paths
image_dir = r"C:\Users\sunil\Downloads\cogniable\frames\val"
label_dir = r"C:\Users\sunil\Downloads\cogniable\labels\val"
os.makedirs(label_dir, exist_ok=True)

# Define the class ID for "person"
PERSON_CLASS_ID = 0

# Process each image
for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)

    # Run detection
    results = model(image)

    # Initialize a flag to check if any person is detected
    person_detected = False

    # Write labels in YOLO format, but only for persons
    label_filename = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(label_dir, label_filename)
    
    with open(label_path, 'w') as f:
        for result in results:
            for bbox, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if int(cls) == PERSON_CLASS_ID:  # Filter for "person" class only
                    person_detected = True
                    x_center = (bbox[0] + bbox[2]) / 2 / image.shape[1]
                    y_center = (bbox[1] + bbox[3]) / 2 / image.shape[0]
                    width = (bbox[2] - bbox[0]) / image.shape[1]
                    height = (bbox[3] - bbox[1]) / image.shape[0]
                    f.write(f'{int(cls)} {x_center} {y_center} {width} {height}\n')

    # If no person was detected, delete the image and the label file
    if not person_detected:
        os.remove(image_path)
        os.remove(label_path)
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

Use the following script to train the YOLOv8 model.



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

Use the following script to perform inference and tracking on a video.



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

### Imporvements:

The current person detection model, built using the YOLOv8m model, did not achieve the desired level of accuracy. One significant factor contributing to this is the computational limitations of my laptop. During the training and inference processes, frequent lag was experienced, and the laptop occasionally shut down, likely due to overheating or insufficient resources to handle the model's computational demands. While the YOLOv8m model is more accurate than lighter versions like YOLOv8 Nano, the performance was hampered by the hardware constraints. 

Additionally, similar issues were encountered with the re-identification (Re-ID) process, where tracking the same person across multiple frames was affected due to these computational limitations. To enhance both detection and Re-ID accuracy, upgrading to more powerful hardware or utilizing cloud-based solutions with higher GPU capabilities would allow the model to run more efficiently, reducing lags and interruptions, and improving overall performance.


### Test-video-outputs
Please find the Output video and the saved custom model in the below drive link
https://drive.google.com/drive/folders/1rc0aFmBm-iuGgJzFNIaBUZ6vphxs7OXq?usp=sharing
