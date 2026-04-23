import cv2
import apriltag
import numpy as np
import freenect
import serial
import time

# Connect to Arduino
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

def get_video():
    array, _ = freenect.sync_get_video()
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

def get_depth():
    depth, _ = freenect.sync_get_depth()
    return depth

def send_command(cmd):
    ser.write(cmd.encode())
    time.sleep(0.05)

detector = apriltag.Detector()

print("Following AprilTag... Press 'q' or ESC to stop, or Ctrl+C")

try:
    while True:
        frame = get_video()
        depth = get_depth()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)
        
        command = 'S'
        status = "No tag - stopped"
        distance_m = 0
        color = (100, 100, 100)
        
        if results:
            r = results[0]
            center_x = int(r.center[0])
            center_y = int(r.center[1])
            
            corners = r.corners.astype(int)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            y1, y2 = max(0, center_y - 5), min(480, center_y + 5)
            x1, x2 = max(0, center_x - 5), min(640, center_x + 5)
            region = depth[y1:y2, x1:x2]
            valid = region[region > 0]
            if len(valid) > 0:
                distance_raw = np.median(valid)
                distance_m = 0.1236 * np.tan(distance_raw / 2842.5 + 1.1863)
            
            cv2.line(frame, (320, 0), (320, 480), (255, 255, 0), 1)
            
            if distance_m > 0 and distance_m < 0.7:
                command = 'S'
                status = "ARRIVED"
                color = (0, 255, 0)
            elif center_x < 280:
                command = 'L'
                status = "TURN LEFT"
                color = (255, 0, 0)
            elif center_x > 360:
                command = 'R'
                status = "TURN RIGHT"
                color = (0, 0, 255)
            else:
                command = 'F'
                status = "FORWARD"
                color = (0, 255, 255)
            
            cv2.putText(frame, f"ID: {r.tag_id}", (center_x - 20, center_y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"{distance_m:.2f}m", (center_x - 20, center_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Dist: {distance_m:.2f}m", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        send_command(command)
        
        cv2.imshow("AprilTag Follower", frame)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            break

except KeyboardInterrupt:
    pass

finally:
    send_command('S')
    print("Stopped")
    ser.close()
    cv2.destroyAllWindows()
    freenect.sync_stop()
