import cv2
import apriltag
import numpy as np
import freenect
import serial
import time

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
SERIAL_TIMEOUT = 0.05

TARGET_TAG_ID = None

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
IMAGE_CENTER_X = FRAME_WIDTH / 2.0

STOP_DISTANCE_M = 0.70
SLOW_DISTANCE_M = 1.20
FAST_DISTANCE_M = 2.00

MIN_FORWARD_PWM = 80
MAX_FORWARD_PWM = 180
DEFAULT_NO_DEPTH_PWM = 80

CENTER_DEADBAND = 0.07
PIVOT_ERROR = 0.82
MIN_ARC_PWM = 25

K_TURN = 85
TURN_SLOWDOWN = 0.45

ADAPTIVE_ERROR_THRESHOLD = 0.45
ADAPTIVE_NOT_IMPROVING_SEC = 0.80
ADAPTIVE_TURN_MULT = 1.35
ADAPTIVE_SPEED_MULT = 0.70
IMPROVEMENT_EPS = 0.025

LEFT_PWM_TRIM = 0
RIGHT_PWM_TRIM = 0

PWM_LIMIT = 160
COMMAND_PERIOD_SEC = 0.10
PRINT_PERIOD_SEC = 0.50

ERROR_SMOOTHING_ALPHA = 0.35
DIST_SMOOTHING_ALPHA = 0.25

TAG_LOST_COAST_SEC = 0.25
TAG_LOST_SCAN_SEC = 4.00
TAG_LOST_STOP_SEC = 6.00
SCAN_PWM = 65
SCAN_DIRECTION_SWITCH_SEC = 1.25
LOST_COAST_MULT = 0.45


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def get_video():
    array, _ = freenect.sync_get_video()
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def get_depth():
    depth, _ = freenect.sync_get_depth()
    return depth


def kinect_raw_to_meters(distance_raw):
    if distance_raw <= 0:
        return 0.0
    return 0.1236 * np.tan(distance_raw / 2842.5 + 1.1863)


def depth_at_tag_center(depth, center_x, center_y):
    y1, y2 = max(0, center_y - 5), min(FRAME_HEIGHT, center_y + 5)
    x1, x2 = max(0, center_x - 5), min(FRAME_WIDTH, center_x + 5)
    region = depth[y1:y2, x1:x2]
    valid = region[region > 0]
    if len(valid) == 0:
        return 0.0
    return float(kinect_raw_to_meters(np.median(valid)))


def tag_area(result):
    corners = result.corners.astype(np.float32)
    return abs(cv2.contourArea(corners))


def choose_tag(results):
    if not results:
        return None

    if TARGET_TAG_ID is not None:
        matches = [r for r in results if r.tag_id == TARGET_TAG_ID]
        if not matches:
            return None
        return max(matches, key=tag_area)

    return max(results, key=tag_area)


def speed_from_distance(distance_m):
    if distance_m <= 0:
        return DEFAULT_NO_DEPTH_PWM
    if distance_m <= STOP_DISTANCE_M:
        return 0
    if distance_m >= FAST_DISTANCE_M:
        return MAX_FORWARD_PWM

    t = (distance_m - STOP_DISTANCE_M) / (FAST_DISTANCE_M - STOP_DISTANCE_M)
    t = clamp(t, 0.0, 1.0)
    return MIN_FORWARD_PWM + t * (MAX_FORWARD_PWM - MIN_FORWARD_PWM)


def compute_motor_pwm(x_error, distance_m, adaptive_active):
    abs_error = abs(x_error)

    if distance_m > 0 and distance_m <= STOP_DISTANCE_M:
        return 0, 0, "ARRIVED"

    base = speed_from_distance(distance_m)

    if abs_error < CENTER_DEADBAND:
        turn = 0.0
        status = "FORWARD"
    else:
        turn = K_TURN * x_error
        status = "ARC RIGHT" if x_error > 0 else "ARC LEFT"

    turn_severity = clamp(abs_error, 0.0, 1.0)
    base *= 1.0 - TURN_SLOWDOWN * turn_severity

    if adaptive_active and abs_error >= ADAPTIVE_ERROR_THRESHOLD:
        turn *= ADAPTIVE_TURN_MULT
        base *= ADAPTIVE_SPEED_MULT
        status = "HARD RIGHT" if x_error > 0 else "HARD LEFT"

    left_pwm = base + turn
    right_pwm = base - turn

    if abs_error < PIVOT_ERROR:
        if left_pwm > 0 or right_pwm > 0:
            left_pwm = max(left_pwm, MIN_ARC_PWM)
            right_pwm = max(right_pwm, MIN_ARC_PWM)

    left_pwm += LEFT_PWM_TRIM
    right_pwm += RIGHT_PWM_TRIM

    left_pwm = int(round(clamp(left_pwm, -PWM_LIMIT, PWM_LIMIT)))
    right_pwm = int(round(clamp(right_pwm, -PWM_LIMIT, PWM_LIMIT)))

    if abs(left_pwm) < 8:
        left_pwm = 0
    if abs(right_pwm) < 8:
        right_pwm = 0

    return left_pwm, right_pwm, status


def compute_scan_pwm(lost_time, last_seen_error, last_visible_pwm):
    if lost_time <= TAG_LOST_COAST_SEC:
        left_pwm = int(last_visible_pwm[0] * LOST_COAST_MULT)
        right_pwm = int(last_visible_pwm[1] * LOST_COAST_MULT)
        return left_pwm, right_pwm, "LOST - COAST"

    if lost_time <= TAG_LOST_SCAN_SEC:
        if last_seen_error is None:
            base_direction = 1
        elif last_seen_error >= 0:
            base_direction = 1
        else:
            base_direction = -1

        scan_phase = int((lost_time - TAG_LOST_COAST_SEC) / SCAN_DIRECTION_SWITCH_SEC)
        if scan_phase % 2 == 1:
            base_direction *= -1

        if base_direction > 0:
            return SCAN_PWM, -SCAN_PWM, "SCAN RIGHT"
        return -SCAN_PWM, SCAN_PWM, "SCAN LEFT"

    if lost_time <= TAG_LOST_STOP_SEC:
        if last_seen_error is not None and last_seen_error >= 0:
            return int(SCAN_PWM * 0.65), int(-SCAN_PWM * 0.65), "SLOW SCAN RIGHT"
        return int(-SCAN_PWM * 0.65), int(SCAN_PWM * 0.65), "SLOW SCAN LEFT"

    return 0, 0, "LOST - STOP"


def send_motor_command(ser, left_pwm, right_pwm):
    packet = f"M,{left_pwm},{right_pwm}\n"
    ser.write(packet.encode("ascii"))


def send_stop(ser):
    ser.write(b"M,0,0\n")


def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
    time.sleep(2.0)

    detector = apriltag.Detector()

    print("Following AprilTag with differential-drive PWM control.")
    print("Serial protocol: M,<left_pwm>,<right_pwm>")

    last_send_time = 0.0
    last_print_time = 0.0
    last_sent = None

    smoothed_error = None
    smoothed_distance = None
    previous_abs_error = None
    not_improving_since = None

    last_seen_time = None
    last_seen_error = None
    last_visible_pwm = (0, 0)

    try:
        while True:
            loop_time = time.time()

            frame = get_video()
            depth = get_depth()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray)
            tag = choose_tag(results)

            left_pwm = 0
            right_pwm = 0
            status = "NO TAG - STOP"
            distance_m = 0.0
            x_error = 0.0
            center_x = None
            center_y = None
            adaptive_active = False

            if tag is not None:
                center_x = int(tag.center[0])
                center_y = int(tag.center[1])

                raw_error = (center_x - IMAGE_CENTER_X) / IMAGE_CENTER_X
                raw_error = clamp(raw_error, -1.0, 1.0)

                measured_distance = depth_at_tag_center(depth, center_x, center_y)

                if smoothed_error is None:
                    smoothed_error = raw_error
                else:
                    smoothed_error = ((1.0 - ERROR_SMOOTHING_ALPHA) * smoothed_error +
                                      ERROR_SMOOTHING_ALPHA * raw_error)
                x_error = smoothed_error

                if measured_distance > 0:
                    if smoothed_distance is None:
                        smoothed_distance = measured_distance
                    else:
                        smoothed_distance = ((1.0 - DIST_SMOOTHING_ALPHA) * smoothed_distance +
                                             DIST_SMOOTHING_ALPHA * measured_distance)
                    distance_m = smoothed_distance
                else:
                    distance_m = 0.0

                abs_error = abs(x_error)
                if abs_error >= ADAPTIVE_ERROR_THRESHOLD:
                    improving = (previous_abs_error is not None and
                                 abs_error < previous_abs_error - IMPROVEMENT_EPS)
                    if improving:
                        not_improving_since = None
                    else:
                        if not_improving_since is None:
                            not_improving_since = loop_time
                        elif loop_time - not_improving_since >= ADAPTIVE_NOT_IMPROVING_SEC:
                            adaptive_active = True
                else:
                    not_improving_since = None

                previous_abs_error = abs_error

                left_pwm, right_pwm, status = compute_motor_pwm(
                    x_error=x_error,
                    distance_m=distance_m,
                    adaptive_active=adaptive_active,
                )

                last_seen_time = loop_time
                last_seen_error = x_error
                last_visible_pwm = (left_pwm, right_pwm)

                # corners = tag.corners.astype(int)
                # cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
                # cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                # cv2.line(frame, (int(IMAGE_CENTER_X), 0), (int(IMAGE_CENTER_X), FRAME_HEIGHT), (255, 255, 0), 1)
                # cv2.putText(frame, f"ID: {tag.tag_id}", (center_x - 20, center_y - 40),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # cv2.putText(frame, f"{distance_m:.2f}m", (center_x - 20, center_y - 15),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                smoothed_error = None
                smoothed_distance = None
                previous_abs_error = None
                not_improving_since = None

                if last_seen_time is None:
                    left_pwm, right_pwm, status = compute_scan_pwm(
                        lost_time=TAG_LOST_COAST_SEC + 0.01,
                        last_seen_error=None,
                        last_visible_pwm=(0, 0),
                    )
                else:
                    lost_time = loop_time - last_seen_time
                    left_pwm, right_pwm, status = compute_scan_pwm(
                        lost_time=lost_time,
                        last_seen_error=last_seen_error,
                        last_visible_pwm=last_visible_pwm,
                    )

            # cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            # cv2.putText(frame, f"Dist: {distance_m:.2f}m", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # cv2.putText(frame, f"PWM L:{left_pwm} R:{right_pwm}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # cv2.imshow("AprilTag Follower", frame)

            command = (left_pwm, right_pwm)
            if command != last_sent or loop_time - last_send_time >= COMMAND_PERIOD_SEC:
                send_motor_command(ser, left_pwm, right_pwm)
                last_sent = command
                last_send_time = loop_time

            if loop_time - last_print_time >= PRINT_PERIOD_SEC:
                if tag is not None:
                    print(
                        f"{status:14s} | tag_id={tag.tag_id} "
                        f"x={center_x:3d} err={x_error:+.2f} "
                        f"dist={distance_m:.2f}m pwm=({left_pwm},{right_pwm}) "
                        f"adaptive={adaptive_active}"
                    )
                else:
                    if last_seen_time is None:
                        print(f"{status:14s} | pwm=({left_pwm},{right_pwm}) lost=never_seen")
                    else:
                        print(f"{status:14s} | pwm=({left_pwm},{right_pwm}) lost={loop_time - last_seen_time:.2f}s")
                last_print_time = loop_time

            time.sleep(0.005)

            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q') or key == 27:
            #     break

    except KeyboardInterrupt:
        pass

    finally:
        print("Stopping motors...")
        try:
            for _ in range(3):
                send_stop(ser)
                time.sleep(0.05)
        finally:
            ser.close()
            # cv2.destroyAllWindows()
            freenect.sync_stop()
        print("Stopped")

if __name__ == "__main__":
    main()
