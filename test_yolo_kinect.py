import freenect
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')

def get_video():
    array, _ = freenect.sync_get_video()
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

print("Running YOLO on Kinect feed... Press 'q' to quit")

while True:
    frame = get_video()
    results = model(frame, verbose=False)
    
    # Draw detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls]} {conf:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("YOLO Kinect", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
freenect.sync_stop()