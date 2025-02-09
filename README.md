# Vehicle Object Detection

This project is a vehicle object detection system that uses deep learning techniques to identify and classify vehicles in images and videos.

---

## How to Run Vehicle Object Detection

### Prerequisites
- Install [Python 3.8+](https://www.python.org/downloads/)
- Install [TensorFlow](https://www.tensorflow.org/install)
- Install [PyTorch](https://pytorch.org/get-started/locally/)
- Install [OpenCV](https://opencv.org/)
- Install [YOLOv7](https://github.com/WongKinYiu/yolov7)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Szh022401/VehicleObjectDetection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd VehicleObjectDetection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Run the Model

### 1. Prepare Data
- Place images or videos in the `input/` folder.

### 2. Run Object Detection
- For image detection:
  ```bash
  python detect.py --source input/image.jpg --weights yolov7.pt --conf 0.5
  ```
- For video detection:
  ```bash
  python detect.py --source input/video.mp4 --weights yolov7.pt --conf 0.5
  ```

### 3. View Results
- The processed images/videos will be saved in the `output/` folder.

---

## Verify System is Running
- Ensure the model initializes correctly without errors.
- Check the `output/` folder for detected objects.

---

## Tech Stack
- **Frameworks**: TensorFlow, PyTorch
- **Model**: YOLOv7
- **Libraries**: OpenCV, NumPy
- **Programming Language**: Python

