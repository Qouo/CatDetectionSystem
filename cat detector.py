import cv2
from datetime import datetime
import time

# Paths to the YOLO model files
weights_path = "Q:\\RA Pictures\\yolov4_new.weights"      
config_path = "Q:\\RA Pictures\\yolov4.cfg"            
classes_file = "Q:\\RA Pictures\\coco.names"

# Load class names
with open(classes_file, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Check if CUDA is available
cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
if cuda_device_count > 0:
    # Force use of the first CUDA device (usually your 3070 will be device 0)
    cv2.cuda.setDevice(0)
    print(f"Using CUDA device: {cv2.cuda.getDeviceName(0)}")
else:
    print("CUDA is not available, using CPU.")

# Get the "cat" class ID
cat_class_id = class_names.index("cat")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Error: Could not access the webcam.")

log_file = open("cat_detection_log.txt", "a")

print("Press 'q' to exit.")

# FPS calculation variables
prev_time = time.time()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video frame.")
        break

    height, width = frame.shape[:2]

    # Resize the frame for faster processing (to a smaller resolution like 416x416)
    resized_frame = cv2.resize(frame, (416, 416))  # Reduce resolution for better FPS

    # Create a blob from the resized frame
    blob = cv2.dnn.blobFromImage(resized_frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    # Parse detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = int(scores.argmax())
            confidence = scores[class_id]

            if class_id == cat_class_id and confidence > 0.5:
                # Get bounding box coordinates
                box = detection[0:4] * [width, height, width, height]
                (center_x, center_y, box_w, box_h) = box.astype("int")
                x = int(center_x - (box_w / 2))
                y = int(center_y - (box_h / 2))

                # Draw bounding box and label
                label = f"Cat: {confidence * 100:.2f}%"
                cv2.rectangle(frame, (x, y), (x + int(box_w), y + int(box_h)), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Log detection
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"Cat detected at {current_time}, Confidence: {confidence * 100:.2f}%\n")
                print(f"Cat detected at {current_time}, Confidence: {confidence * 100:.2f}%")

    # Calculate FPS
    current_time = time.time()
    elapsed_time = current_time - prev_time
    if elapsed_time > 0:
        fps = int(1 / elapsed_time)

    prev_time = current_time

    # Display FPS and exit message on the screen
    cv2.putText(frame, f"FPS: {fps}", (width - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame with bounding boxes around detected cats
    cv2.imshow("Cat Detector", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
log_file.close()
