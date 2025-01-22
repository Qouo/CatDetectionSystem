# Cat Detection Project

**A real-time cat detection system using YOLOv4.**

This project leverages the power of the YOLOv4 deep learning model to accurately and efficiently detect cats in real-time video streams. 

## Key Features

* **Real-time Cat Detection:** Utilizes the YOLOv4 object detection algorithm for swift and reliable cat identification.
* **Bounding Boxes:** Draws precise bounding boxes around detected cats with associated confidence scores.
* **FPS Counter:** Displays real-time frames per second (FPS) for performance monitoring.
* **Detection Logging:** Logs each cat detection event with timestamp and confidence score for further analysis.

## Installation

**Prerequisites**

* Python 3.x
* OpenCV with DNN module (`pip install opencv-python`)
* YOLOv4 weights, configuration, and class files (e.g., `coco.names`)
* (Optional) CUDA for GPU acceleration

**Dependencies**

```bash
pip install opencv-python opencv-python-headless
Obtain YOLOv4 Files

Download the following essential files for the YOLOv4 model:

yolov4.cfg: The configuration file defining the YOLOv4 network architecture.
yolov4.weights: Pre-trained weights for the YOLOv4 model.
coco.names: A text file containing the list of object classes, including "cat."
Place these files in the appropriate directories and update the weights_path, config_path, and classes_file variables in the code accordingly.

Usage
Run the Script: Execute the Python script using the following command:

Bash

python cat_detection.py
View Results: The program will display a live video feed with bounding boxes drawn around detected cats.
Real-time FPS and detection information will be displayed on the video stream.

Exit: Press the 'q' key on your keyboard to exit the program.

Detection Log:

Each successful cat detection will be logged to a file named cat_detection_log.txt, containing the timestamp and confidence score for each detection.

Optimization Plan
Resolution Optimization: Adjust the frame resolution for improved performance on resource-constrained devices.
Hardware Acceleration: Leverage GPU acceleration (CUDA) for significant speedup.
Model Quantization: Explore model quantization techniques to reduce model size and improve inference speed.
End Goal: Raspberry Pi Deployment
Headless Operation: Deploy the system on a headless Raspberry Pi for continuous, unattended operation.
Telegram Notifications: Send real-time notifications with images of detected cats to a designated Telegram account.
Optimized Performance: Fine-tune the system for optimal performance and resource utilization on the Raspberry Pi platform.
Contributing
Contributions are always welcome! Feel free to fork this repository, submit issues, or submit pull requests.

License
This project is licensed under the MIT License.

Note:

This project is currently designed for use with a webcam. For Raspberry Pi deployment, modifications will be necessary to enable headless operation and optimize performance.

