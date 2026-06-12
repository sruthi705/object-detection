!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install opencv-python
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# Use a sample image from the web
img_path = 'https://tse1.mm.bing.net/th/id/OIP.z5ifNsidzpeXtJR79yCjsgHaE7?rs=1&pid=ImgDetMain'

# Run detection
results = model(img_path)

# Print results
results.print()
# Display the image with bounding boxes
results.show()
import cv2
import numpy as np

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Render results on the frame
    result_img = np.squeeze(results.render())

    # Show the frame
    cv2.imshow("YOLOv5 Webcam Detection", result_img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()
