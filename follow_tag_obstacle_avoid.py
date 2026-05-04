import cv2
import apriltag
import numpy as np
import freenect
import serial
import time
import heapq
import math

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
SERIAL_TIMEOUT = 0.05

TARGET_TAG_ID = None

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
IMAGE_CENTER_X = FRAME_WIDTH / 2.0
IMAGE_CENTER_Y = FRAME_HEIGHT / 2.0

KINECT_HFOV_DEG = 57.0
KINECT_VFOV_DEG = 43.0
FX = (FRAME_WIDTH / 2.0) / math.tan(math.radians(KINECT_HFOV_DEG / 2.0))
FY = (FRAME_HEIGHT / 2.0) / math.tan(math.radians(KINECT_VFOV_DEG / 2.0))

KINECT_HEIGHT_M = 0.60
KINECT_PITCH_DOWN_DEG = 8.0
MIN_OBSTACLE_HEIGHT_M = 0.08
MAX_OBSTACLE_HEIGHT_M = 1.20

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

CART_WIDTH_M = 0.55
CLEARANCE_MARGIN_M = 0.15
ROBOT_RADIUS_M = CART_WIDTH_M / 2.0 + CLEARANCE_MARGIN_M

GRID_X_MIN_M = -2.2
GRID_X_MAX_M = 2.2
GRID_Z_MIN_M = 0.15
GRID_Z_MAX_M = 4.0
GRID_RES_M = 0.15

DEPTH_ROI_Y_START = int(FRAME_HEIGHT * 0.45)
DEPTH_ROI_Y_END = int(FRAME_HEIGHT * 0.85)
DEPTH_BLOCK_SIZE = 12
MIN_VALID_DEPTH_M = 0.25
MAX_VALID_DEPTH_M = 4.0
MIN_BLOCK_VALID_PIXELS = 10

DIRECT_PATH_LATERAL_STEP_M = 0.10
PATH_LOOKAHEAD_M = 0.80
PLANNER_MAX_TARGET_Z_M = 3.5
PLANNER_TARGET_BACKOFF_M = 0.35

LOST_COAST_SEC = 0.25
LOST_SEARCH_SEC = 1.50
LOST_STOP_SEC = 3.00
LOST_COAST_MULT = 0.45
SEARCH_PWM = 65

FRONT_STOP_DISTANCE_M = 0.55
FRONT_STOP_HALF_WIDTH_M = 0.35
FRONT_STOP_MIN_POINTS = 6

OBSTACLE_STOP_HOLD_SEC = 0.35


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


def raw_depth_array_to_meters(depth_raw):
    depth_float = depth_raw.astype(np.float32)
    valid = depth_float > 0
    depth_m = np.zeros_like(depth_float, dtype=np.float32)
    depth_m[valid] = 0.1236 * np.tan(depth_float[valid] / 2842.5 + 1.1863)
    return depth_m


def depth_at_tag_center(depth, center_x, center_y):
    y1, y2 = max(0, center_y - 5), min(FRAME_HEIGHT, center_y + 5)
    x1, x2 = max(0, center_x - 5), min(FRAME_WIDTH, center_x + 5)
    region = depth[y1:y2, x1:x2]
    valid = region[region > 0]
    if len(valid) == 0:
        return 0.0
    return float(kinect_raw_to_meters(np.median(valid)))


def project_pixel_to_camera_xyz(u, v, depth_m):
    x = ((float(u) - IMAGE_CENTER_X) / FX) * depth_m
    y_down = ((float(v) - IMAGE_CENTER_Y) / FY) * depth_m
    z = depth_m
    return x, y_down, z


def camera_y_down_to_height(y_down, z):
    pitch = math.radians(KINECT_PITCH_DOWN_DEG)
    world_down = math.cos(pitch) * y_down + math.sin(pitch) * z
    return KINECT_HEIGHT_M - world_down


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


def compute_motor_pwm(x_error, distance_m, adaptive_active, speed_scale=1.0):
    abs_error = abs(x_error)

    if distance_m > 0 and distance_m <= STOP_DISTANCE_M:
        return 0, 0, "ARRIVED"

    base = speed_from_distance(distance_m) * speed_scale

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


def send_motor_command(ser, left_pwm, right_pwm):
    packet = f"M,{left_pwm},{right_pwm}\n"
    ser.write(packet.encode("ascii"))


def send_stop(ser):
    ser.write(b"M,0,0\n")


def grid_shape():
    width = int(round((GRID_X_MAX_M - GRID_X_MIN_M) / GRID_RES_M)) + 1
    height = int(round((GRID_Z_MAX_M - GRID_Z_MIN_M) / GRID_RES_M)) + 1
    return height, width


def xz_to_grid(x, z):
    gx = int(round((x - GRID_X_MIN_M) / GRID_RES_M))
    gz = int(round((z - GRID_Z_MIN_M) / GRID_RES_M))
    return gz, gx


def grid_to_xz(gz, gx):
    x = GRID_X_MIN_M + gx * GRID_RES_M
    z = GRID_Z_MIN_M + gz * GRID_RES_M
    return x, z


def in_grid(gz, gx, shape):
    return 0 <= gz < shape[0] and 0 <= gx < shape[1]


def add_occupied_with_inflation(occupied, x, z):
    gz, gx = xz_to_grid(x, z)
    radius_cells = int(math.ceil(ROBOT_RADIUS_M / GRID_RES_M))
    shape = occupied.shape

    for dz in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            ngz = gz + dz
            ngx = gx + dx
            if not in_grid(ngz, ngx, shape):
                continue
            if math.hypot(dx * GRID_RES_M, dz * GRID_RES_M) <= ROBOT_RADIUS_M:
                occupied[ngz, ngx] = True


def build_free_space_grid(depth_m):
    shape = grid_shape()
    occupied = np.zeros(shape, dtype=bool)
    obstacle_points = []
    raw_candidate_count = 0
    height_rejected_count = 0

    for y in range(DEPTH_ROI_Y_START, DEPTH_ROI_Y_END, DEPTH_BLOCK_SIZE):
        for x in range(0, FRAME_WIDTH, DEPTH_BLOCK_SIZE):
            block = depth_m[y:y + DEPTH_BLOCK_SIZE, x:x + DEPTH_BLOCK_SIZE]
            valid = block[(block >= MIN_VALID_DEPTH_M) & (block <= MAX_VALID_DEPTH_M)]
            if valid.size < MIN_BLOCK_VALID_PIXELS:
                continue

            z = float(np.median(valid))
            u = x + DEPTH_BLOCK_SIZE / 2.0
            v = y + DEPTH_BLOCK_SIZE / 2.0
            px, py_down, pz = project_pixel_to_camera_xyz(u, v, z)
            point_height = camera_y_down_to_height(py_down, pz)
            raw_candidate_count += 1

            if not (MIN_OBSTACLE_HEIGHT_M <= point_height <= MAX_OBSTACLE_HEIGHT_M):
                height_rejected_count += 1
                continue

            if GRID_X_MIN_M <= px <= GRID_X_MAX_M and GRID_Z_MIN_M <= pz <= GRID_Z_MAX_M:
                add_occupied_with_inflation(occupied, px, pz)
                obstacle_points.append((px, pz, point_height))

    return occupied, obstacle_points, raw_candidate_count, height_rejected_count


def front_obstacle_too_close(obstacle_points):
    count = 0
    for x, z, height in obstacle_points:
        if abs(x) <= FRONT_STOP_HALF_WIDTH_M and 0.0 < z <= FRONT_STOP_DISTANCE_M:
            count += 1
            if count >= FRONT_STOP_MIN_POINTS:
                return True
    return False


def line_is_free(occupied, x0, z0, x1, z1):
    distance = math.hypot(x1 - x0, z1 - z0)
    steps = max(2, int(distance / DIRECT_PATH_LATERAL_STEP_M))
    shape = occupied.shape

    for i in range(steps + 1):
        t = i / float(steps)
        x = x0 + t * (x1 - x0)
        z = z0 + t * (z1 - z0)
        gz, gx = xz_to_grid(x, z)
        if not in_grid(gz, gx, shape):
            return False
        if occupied[gz, gx]:
            return False
    return True


def nearest_free_cell(occupied, desired_x, desired_z, max_radius_cells=10):
    start_gz, start_gx = xz_to_grid(desired_x, desired_z)
    shape = occupied.shape

    if in_grid(start_gz, start_gx, shape) and not occupied[start_gz, start_gx]:
        return start_gz, start_gx

    for r in range(1, max_radius_cells + 1):
        best = None
        best_dist = float("inf")
        for dz in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) != r and abs(dz) != r:
                    continue
                gz = start_gz + dz
                gx = start_gx + dx
                if not in_grid(gz, gx, shape) or occupied[gz, gx]:
                    continue
                x, z = grid_to_xz(gz, gx)
                d = math.hypot(x - desired_x, z - desired_z)
                if d < best_dist:
                    best = (gz, gx)
                    best_dist = d
        if best is not None:
            return best

    return None


def astar_path(occupied, start, goal):
    shape = occupied.shape
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
    ]

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
    came_from = {}
    cost_so_far = {start: 0.0}

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dz, dx, step_cost in neighbors:
            ngz = current[0] + dz
            ngx = current[1] + dx
            nxt = (ngz, ngx)

            if not in_grid(ngz, ngx, shape):
                continue
            if occupied[ngz, ngx]:
                continue

            extra = 0.0
            x, z = grid_to_xz(ngz, ngx)
            extra += 0.08 * abs(x)
            if z < GRID_Z_MIN_M + 0.2:
                extra += 1.0

            new_cost = current_cost + step_cost + extra
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + heuristic(nxt, goal)
                heapq.heappush(open_heap, (priority, new_cost, nxt))
                came_from[nxt] = current

    return None


def choose_lookahead_waypoint(path):
    if not path:
        return None

    accumulated = 0.0
    prev_x, prev_z = grid_to_xz(path[0][0], path[0][1])

    for cell in path[1:]:
        x, z = grid_to_xz(cell[0], cell[1])
        accumulated += math.hypot(x - prev_x, z - prev_z)
        if accumulated >= PATH_LOOKAHEAD_M:
            return x, z
        prev_x, prev_z = x, z

    return grid_to_xz(path[-1][0], path[-1][1])


def plan_path_to_tag(occupied, tag_x_m, tag_z_m):
    target_z = clamp(tag_z_m - PLANNER_TARGET_BACKOFF_M, GRID_Z_MIN_M + 0.2, PLANNER_MAX_TARGET_Z_M)
    target_x = clamp(tag_x_m, GRID_X_MIN_M + ROBOT_RADIUS_M, GRID_X_MAX_M - ROBOT_RADIUS_M)

    if line_is_free(occupied, 0.0, GRID_Z_MIN_M, target_x, target_z):
        lookahead_z = min(PATH_LOOKAHEAD_M, target_z)
        lookahead_x = target_x * (lookahead_z / max(target_z, 0.01))
        return {
            "found": True,
            "mode": "DIRECT",
            "waypoint_x": lookahead_x,
            "waypoint_z": lookahead_z,
            "target_x": target_x,
            "target_z": target_z,
            "path_len": 2,
        }

    start = nearest_free_cell(occupied, 0.0, GRID_Z_MIN_M + 0.10, max_radius_cells=6)
    goal = nearest_free_cell(occupied, target_x, target_z, max_radius_cells=12)

    if start is None or goal is None:
        return {"found": False, "mode": "NO_FREE_START_OR_GOAL"}

    path = astar_path(occupied, start, goal)
    if path is None:
        return {"found": False, "mode": "NO_PATH"}

    waypoint = choose_lookahead_waypoint(path)
    if waypoint is None:
        return {"found": False, "mode": "NO_WAYPOINT"}

    return {
        "found": True,
        "mode": "PLANNED",
        "waypoint_x": waypoint[0],
        "waypoint_z": waypoint[1],
        "target_x": target_x,
        "target_z": target_z,
        "path_len": len(path),
    }


def xz_to_error(x, z):
    if z <= 0:
        return 0.0
    angle = math.atan2(x, z)
    max_angle = math.radians(KINECT_HFOV_DEG / 2.0)
    return clamp(angle / max_angle, -1.0, 1.0)


def compute_search_pwm(last_error):
    if last_error is None:
        return 0, 0, "LOST - STOP"

    if last_error >= 0:
        return SEARCH_PWM, -SEARCH_PWM, "SEARCH RIGHT"
    return -SEARCH_PWM, SEARCH_PWM, "SEARCH LEFT"


def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
    time.sleep(2.0)

    detector = apriltag.Detector()

    print("Following AprilTag with depth-grid free-space path planning.")
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
    last_plan = None
    obstacle_stop_until = 0.0

    try:
        while True:
            loop_time = time.time()

            frame = get_video()
            depth = get_depth()
            depth_m = raw_depth_array_to_meters(depth)
            occupied, obstacle_points, raw_candidates, height_rejected = build_free_space_grid(depth_m)
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
            plan_mode = "NONE"
            path_len = 0

            occupied_ratio = float(np.mean(occupied))

            if front_obstacle_too_close(obstacle_points):
                obstacle_stop_until = loop_time + OBSTACLE_STOP_HOLD_SEC

            obstacle_blocking = loop_time < obstacle_stop_until

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

                tag_z_m = distance_m if distance_m > 0 else DEFAULT_NO_DEPTH_PWM / float(MAX_FORWARD_PWM)
                tag_x_m = x_error * tag_z_m * math.tan(math.radians(KINECT_HFOV_DEG / 2.0))

                last_seen_time = loop_time
                last_seen_error = x_error

                if distance_m > 0 and distance_m <= STOP_DISTANCE_M:
                    left_pwm, right_pwm, status = 0, 0, "ARRIVED"
                    last_plan = None
                elif obstacle_blocking:
                    left_pwm, right_pwm, status = 0, 0, "OBSTACLE STOP"
                else:
                    plan = plan_path_to_tag(occupied, tag_x_m, tag_z_m)
                    last_plan = plan
                    plan_mode = plan.get("mode", "NONE")
                    path_len = plan.get("path_len", 0)

                    if plan.get("found"):
                        waypoint_error = xz_to_error(plan["waypoint_x"], plan["waypoint_z"])
                        waypoint_distance = min(distance_m if distance_m > 0 else FAST_DISTANCE_M, plan["waypoint_z"] + 0.4)

                        abs_error = abs(waypoint_error)
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

                        speed_scale = 0.85 if plan_mode == "PLANNED" else 1.0
                        left_pwm, right_pwm, status = compute_motor_pwm(
                            x_error=waypoint_error,
                            distance_m=waypoint_distance,
                            adaptive_active=adaptive_active,
                            speed_scale=speed_scale,
                        )
                        status = f"{plan_mode} {status}"
                        x_error = waypoint_error
                    else:
                        left_pwm, right_pwm, status = 0, 0, f"NO PATH {plan_mode}"

                last_visible_pwm = (left_pwm, right_pwm)

            else:
                smoothed_error = None
                smoothed_distance = None
                previous_abs_error = None
                not_improving_since = None

                if last_seen_time is None:
                    left_pwm, right_pwm, status = 0, 0, "NO TAG - STOP"
                else:
                    lost_time = loop_time - last_seen_time
                    if obstacle_blocking:
                        left_pwm, right_pwm, status = 0, 0, "LOST + OBSTACLE STOP"
                    elif lost_time <= LOST_COAST_SEC:
                        left_pwm = int(last_visible_pwm[0] * LOST_COAST_MULT)
                        right_pwm = int(last_visible_pwm[1] * LOST_COAST_MULT)
                        status = "LOST - COAST"
                    elif lost_time <= LOST_SEARCH_SEC:
                        left_pwm, right_pwm, status = compute_search_pwm(last_seen_error)
                    elif lost_time <= LOST_STOP_SEC:
                        left_pwm, right_pwm, status = compute_search_pwm(last_seen_error)
                        left_pwm = int(left_pwm * 0.65)
                        right_pwm = int(right_pwm * 0.65)
                        status = "LOST - SLOW SEARCH"
                    else:
                        left_pwm, right_pwm, status = 0, 0, "LOST - STOP"

            command = (left_pwm, right_pwm)
            if command != last_sent or loop_time - last_send_time >= COMMAND_PERIOD_SEC:
                send_motor_command(ser, left_pwm, right_pwm)
                last_sent = command
                last_send_time = loop_time

            if loop_time - last_print_time >= PRINT_PERIOD_SEC:
                if tag is not None:
                    print(
                        f"{status:20s} | tag_id={tag.tag_id} "
                        f"x={center_x:3d} err={x_error:+.2f} "
                        f"dist={distance_m:.2f}m pwm=({left_pwm},{right_pwm}) "
                        f"plan={plan_mode} path={path_len} occ={occupied_ratio:.2f} "
                        f"cand={raw_candidates} rej={height_rejected} obst={obstacle_blocking}"
                    )
                else:
                    print(
                        f"{status:20s} | pwm=({left_pwm},{right_pwm}) "
                        f"occ={occupied_ratio:.2f} cand={raw_candidates} rej={height_rejected} "
                        f"obst={obstacle_blocking}"
                    )
                last_print_time = loop_time

            time.sleep(0.005)

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
            freenect.sync_stop()
        print("Stopped")


if __name__ == "__main__":
    main()
