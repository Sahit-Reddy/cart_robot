import cv2
import apriltag
import numpy as np
import freenect

def get_video():
    array, _ = freenect.sync_get_video()
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

def get_depth():
    depth, _ = freenect.sync_get_depth()
    return depth

detector = apriltag.Detector()

print("Press 'q' to quit")

while True:
    frame = get_video()
    depth = get_depth()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)
    
    for r in results:
        # Draw box around tag
        corners = r.corners.astype(int)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        
        # Get center point
        center_x, center_y = int(r.center[0]), int(r.center[1])
        center = (center_x, center_y)
        
        # Sample a 10x10 region around center
        y1, y2 = max(0, center_y - 5), min(480, center_y + 5)
        x1, x2 = max(0, center_x - 5), min(640, center_x + 5)
        region = depth[y1:y2, x1:x2]
        valid = region[region > 0]
        if len(valid) > 0:
            distance_raw = np.median(valid)
        else:
            distance_raw = 0
        
        # Convert to meters (Kinect v1 raw depth formula)
        if distance_raw > 0:
            distance_m = 0.1236 * np.tan(distance_raw / 2842.5 + 1.1863)
        else:
            distance_m = 0
        
        # Draw center point
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        # Draw tag ID and distance
        cv2.putText(frame, f"ID: {r.tag_id}", (center_x - 20, center_y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"{distance_m:.2f}m", (center_x - 20, center_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("AprilTag Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
freenect.sync_stop()
