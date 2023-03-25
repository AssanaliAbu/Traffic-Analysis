import cv2
import torch
import argparse
import numpy as np
from tensorflow.keras.models import load_model

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_video', type=str, required=True, help='Path to input video file')
parser.add_argument('--output_video', type=str, required=True, help='Path to output video file')
# parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv5 weights file')
parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold for object detection')
parser.add_argument('--car_type_model', type=str, required=True, help='Path to car type detection model')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
args = parser.parse_args()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(args.device).eval()
# Load car type detection model
car_type_model = load_model(args.car_type_model)

# Open input and output video streams
cap = cv2.VideoCapture(args.input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))




# Object detection loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame, size=640)
    detections = results.xyxy[0]

    # Filter detections by class and confidence threshold
    cars = detections[(detections[:, 5] == 2) & (detections[:, 4] >= args.conf_threshold)]

    # Draw bounding boxes on frame
    for car in cars:
        x1, y1, x2, y2, conf, cls = car
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, f'{conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 # Extract car image from frame and resize to match car type detection model input shape
        car_image = frame[int(y1):int(y2), int(x1):int(x2)]
        car_image = cv2.resize(car_image, (224, 224))

        # Normalize image pixel values
        car_image = car_image / 255.0

        # Make car type prediction
        # car_type_probs = car_type_model.predict(np.expand_dims(car_image, axis=0))[0]
        # car_type = np.argmax(car_type_probs)
        car_type_probs = car_type_model.predict(car_image)[0]
        car_type = np.argmax(car_type_probs)
  
        # Add car type label to bounding box
        label = 'Car Type: {}'.format(car_type)
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # Write output frame
    out.write(frame)

    # Display output frame (optional)
    # cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video streams
cap.release()