import freenect
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')

# Frame dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Obstacle zone - center of frame where cart is heading
ZONE_LEFT = 200
ZONE_RIGHT = 440

# Only stop for these obstacle classes
OBSTACLE_CLASSES = [0]  # 0 = person

def get_video():
    array, _ = freenect.sync_get_video()
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

def check_obstacle(results):
    """Check if any detection is in the cart's path"""
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:  # Ignore low confidence
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            
            if cls in OBSTACLE_CLASSES:
                center_x = (x1 + x2) / 2
                if ZONE_LEFT < center_x < ZONE_RIGHT:
                    return True, model.names[cls]
    return False, None

print("Running YOLO obstacle detection... Press 'q' to quit")

while True:
    frame = get_video()
    results = model(frame, verbose=False, imgsz=320)
    
    obstacle, obj_name = check_obstacle(results)
    
    # Draw the danger zone
    cv2.rectangle(frame, (ZONE_LEFT, 0), (ZONE_RIGHT, FRAME_HEIGHT), (0, 255, 255), 2)
    
    # Draw detections
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            
            color = (0, 0, 255) if obstacle else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display status
    if obstacle:
        status = f"STOP - {obj_name} in path"
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "CLEAR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    cv2.imshow("YOLO Obstacle Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
freenect.sync_stop()