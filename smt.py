import torch
import time
import cv2

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# Images
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
cap = cv2.VideoCapture(r"D:\IC-Lab\yolor\1.mp4")
while True:
    t = time.time()
    ret, img = cap.read()
    results = model(img)
    print(results)
    # Results
    results.show()    # or .show(), .save(), .crop(), .pandas(), etc.
    print(1 // (time.time() - t))