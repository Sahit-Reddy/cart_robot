#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from cv_bridge import CvBridge


class DepthFreespaceViewer(Node):
    def __init__(self):
        super().__init__("depth_freespace_viewer")
        self.bridge = CvBridge()

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("tag_pose_topic", "/tag_pose")
        self.declare_parameter("tag_distance_topic", "/tag_distance")

        self.declare_parameter("hz", 15.0)

        self.declare_parameter("roi_x_min_frac", 0.05)
        self.declare_parameter("roi_x_max_frac", 0.95)
        self.declare_parameter("roi_y_min_frac", 0.35)
        self.declare_parameter("roi_y_max_frac", 1.00)

        self.declare_parameter("depth_stride", 2)
        self.declare_parameter("depth_limit_mode", "fixed")
        self.declare_parameter("fixed_depth_limit_m", 3.0)
        self.declare_parameter("tag_depth_buffer_m", 0.08)
        self.declare_parameter("min_valid_depth_m", 0.12)

        self.declare_parameter("camera_height_m", 0.362)
        self.declare_parameter("ground_height_tolerance_m", 0.055)
        self.declare_parameter("near_ground_ignore_m", 0.015)
        self.declare_parameter("max_obstacle_height_m", 0.55)

        self.declare_parameter("floor_seed_bottom_frac", 0.18)
        self.declare_parameter("ransac_iters", 80)
        self.declare_parameter("ransac_sample_count", 3)
        self.declare_parameter("ransac_inlier_thresh_m", 0.035)
        self.declare_parameter("ransac_min_inlier_ratio", 0.18)
        self.declare_parameter("ransac_max_points", 5000)

        self.declare_parameter("vertical_consistency_px", 6)
        self.declare_parameter("vertical_same_depth_eps_m", 0.020)
        self.declare_parameter("vertical_large_jump_m", 0.18)

        self.declare_parameter("morph_open_px", 3)
        self.declare_parameter("morph_close_px", 9)
        self.declare_parameter("min_free_blob_area_px", 800)

        self.declare_parameter("show_topdown", True)
        self.declare_parameter("topdown_width_m", 1.4)
        self.declare_parameter("topdown_forward_m", 3.0)
        self.declare_parameter("topdown_resolution_m", 0.025)

        self.rgb: Optional[np.ndarray] = None
        self.depth_m: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None
        self.tag_distance_m: Optional[float] = None

        self.create_subscription(Image, self.get_parameter("rgb_topic").value, self.rgb_cb, 5)
        self.create_subscription(Image, self.get_parameter("depth_topic").value, self.depth_cb, 5)
        self.create_subscription(CameraInfo, self.get_parameter("camera_info_topic").value, self.info_cb, 5)
        self.create_subscription(PoseStamped, self.get_parameter("tag_pose_topic").value, self.tag_pose_cb, 5)
        self.create_subscription(Float32, self.get_parameter("tag_distance_topic").value, self.tag_distance_cb, 5)

        hz = float(self.get_parameter("hz").value)
        self.timer = self.create_timer(1.0 / max(hz, 0.1), self.tick)
        self.get_logger().info("Depth freespace viewer ready")

    def rgb_cb(self, msg: Image):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def depth_cb(self, msg: Image):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if d.dtype == np.uint16:
            self.depth_m = d.astype(np.float32) / 1000.0
        else:
            self.depth_m = d.astype(np.float32)

    def info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)

    def tag_pose_cb(self, msg: PoseStamped):
        x = float(msg.pose.position.x)
        z = float(msg.pose.position.z)
        if z > 0.01:
            self.tag_distance_m = math.sqrt(x * x + z * z)

    def tag_distance_cb(self, msg: Float32):
        self.tag_distance_m = max(0.0, float(msg.data))

    def crop_box(self, shape) -> Tuple[int, int, int, int]:
        h, w = shape[:2]
        x0 = int(w * float(self.get_parameter("roi_x_min_frac").value))
        x1 = int(w * float(self.get_parameter("roi_x_max_frac").value))
        y0 = int(h * float(self.get_parameter("roi_y_min_frac").value))
        y1 = int(h * float(self.get_parameter("roi_y_max_frac").value))
        x0, x1 = np.clip([x0, x1], 0, w)
        y0, y1 = np.clip([y0, y1], 0, h)
        return int(x0), int(y0), int(x1), int(y1)

    def depth_limit(self) -> float:
        fixed = float(self.get_parameter("fixed_depth_limit_m").value)
        mode = str(self.get_parameter("depth_limit_mode").value)
        if mode == "tag" and self.tag_distance_m is not None:
            return max(0.1, self.tag_distance_m - float(self.get_parameter("tag_depth_buffer_m").value))
        if mode == "min_tag_fixed" and self.tag_distance_m is not None:
            return max(0.1, min(fixed, self.tag_distance_m - float(self.get_parameter("tag_depth_buffer_m").value)))
        return fixed

    def project_roi_to_points(self, depth_roi: np.ndarray, box: Tuple[int, int, int, int]):
        x0, y0, _, _ = box
        stride = int(self.get_parameter("depth_stride").value)
        stride = max(1, stride)

        h, w = depth_roi.shape[:2]
        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        z = depth_roi[0:h:stride, 0:w:stride]

        zmin = float(self.get_parameter("min_valid_depth_m").value)
        zmax = self.depth_limit()
        valid = np.isfinite(z) & (z > zmin) & (z < zmax)

        if self.K is None:
            return None

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        u_full = uu.astype(np.float32) + float(x0)
        v_full = vv.astype(np.float32) + float(y0)

        x = (u_full - cx) * z / fx
        y_down = (v_full - cy) * z / fy
        pts = np.stack([x, y_down, z], axis=-1)
        return pts, valid, uu, vv

    def fit_ground_plane(self, pts: np.ndarray, valid: np.ndarray):
        h = pts.shape[0]
        bottom_frac = float(self.get_parameter("floor_seed_bottom_frac").value)
        seed_start = int(max(0, h * (1.0 - bottom_frac)))
        seed_mask = valid.copy()
        seed_mask[:seed_start, :] = False

        candidates = pts[seed_mask]
        candidates = candidates[np.isfinite(candidates).all(axis=1)]
        if candidates.shape[0] < 50:
            candidates = pts[valid]
            candidates = candidates[np.isfinite(candidates).all(axis=1)]
        if candidates.shape[0] < 50:
            return None, None

        max_points = int(self.get_parameter("ransac_max_points").value)
        if candidates.shape[0] > max_points:
            idx = np.random.choice(candidates.shape[0], max_points, replace=False)
            candidates = candidates[idx]

        iters = int(self.get_parameter("ransac_iters").value)
        thresh = float(self.get_parameter("ransac_inlier_thresh_m").value)
        min_ratio = float(self.get_parameter("ransac_min_inlier_ratio").value)

        best_n = None
        best_d = None
        best_count = 0

        for _ in range(iters):
            idx = np.random.choice(candidates.shape[0], 3, replace=False)
            p1, p2, p3 = candidates[idx]
            n = np.cross(p2 - p1, p3 - p1)
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            n = n / norm
            if n[1] < 0:
                n = -n
            d = -float(np.dot(n, p1))
            dist = np.abs(candidates @ n + d)
            count = int(np.sum(dist < thresh))
            if count > best_count:
                best_count = count
                best_n = n
                best_d = d

        if best_n is None:
            return None, None

        if best_count / max(1, candidates.shape[0]) < min_ratio:
            return None, None

        all_pts = pts[valid]
        dist = np.abs(all_pts @ best_n + best_d)
        inliers = all_pts[dist < thresh]
        if inliers.shape[0] >= 20:
            centroid = inliers.mean(axis=0)
            _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
            n = vh[-1]
            if n[1] < 0:
                n = -n
            d = -float(np.dot(n, centroid))
            return n.astype(np.float32), float(d)

        return best_n.astype(np.float32), float(best_d)

    def classify_freespace(self, depth_roi: np.ndarray, box: Tuple[int, int, int, int]):
        projected = self.project_roi_to_points(depth_roi, box)
        if projected is None:
            return None, None
        pts, valid_small, uu, vv = projected

        plane_n, plane_d = self.fit_ground_plane(pts, valid_small)
        h, w = depth_roi.shape[:2]
        free_full = np.zeros((h, w), dtype=np.uint8)
        obstacle_full = np.zeros((h, w), dtype=np.uint8)
        unknown_full = np.ones((h, w), dtype=np.uint8)

        if plane_n is None:
            return self.vertical_depth_fallback(depth_roi), None

        signed = pts @ plane_n + plane_d
        abs_dist = np.abs(signed)

        ground_tol = float(self.get_parameter("ground_height_tolerance_m").value)
        near_ground_ignore = float(self.get_parameter("near_ground_ignore_m").value)
        max_obs_h = float(self.get_parameter("max_obstacle_height_m").value)

        floor_like_small = valid_small & (abs_dist < ground_tol)
        obstacle_small = valid_small & (signed < -near_ground_ignore) & (np.abs(signed) < max_obs_h)
        obstacle_small |= self.vertical_obstacle_cues(depth_roi, uu, vv, valid_small)

        free_full[vv[floor_like_small], uu[floor_like_small]] = 255
        obstacle_full[vv[obstacle_small], uu[obstacle_small]] = 255
        unknown_full[vv[valid_small], uu[valid_small]] = 0

        free_full = self.expand_sampled_mask(free_full)
        obstacle_full = self.expand_sampled_mask(obstacle_full)
        unknown_full = self.expand_sampled_mask(unknown_full)

        free_full[obstacle_full > 0] = 0
        free_full = self.clean_mask(free_full)
        free_full = self.keep_large_blobs(free_full)

        return free_full, {
            "plane_n": plane_n,
            "plane_d": plane_d,
            "obstacle": obstacle_full,
            "unknown": unknown_full,
        }

    def vertical_obstacle_cues(self, depth_roi, uu, vv, valid_small):
        z = depth_roi
        stride = int(self.get_parameter("depth_stride").value)
        same_eps = float(self.get_parameter("vertical_same_depth_eps_m").value)
        large_jump = float(self.get_parameter("vertical_large_jump_m").value)
        run = int(self.get_parameter("vertical_consistency_px").value)

        sampled_z = z[0:z.shape[0]:stride, 0:z.shape[1]:stride]
        dz = np.full_like(sampled_z, np.nan, dtype=np.float32)
        dz[1:, :] = sampled_z[:-1, :] - sampled_z[1:, :]

        same = np.isfinite(dz) & (np.abs(dz) < same_eps)
        jump = np.isfinite(dz) & (np.abs(dz) > large_jump)

        if run > 1:
            kernel = np.ones((run, 1), dtype=np.uint8)
            same_run = cv2.filter2D(same.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_REPLICATE) >= run
        else:
            same_run = same

        cues = (same_run | jump) & valid_small
        return cues

    def expand_sampled_mask(self, mask: np.ndarray):
        stride = int(self.get_parameter("depth_stride").value)
        if stride <= 1:
            return mask
        kernel = np.ones((stride + 1, stride + 1), dtype=np.uint8)
        return cv2.dilate(mask, kernel)

    def clean_mask(self, mask: np.ndarray):
        open_px = int(self.get_parameter("morph_open_px").value)
        close_px = int(self.get_parameter("morph_close_px").value)

        out = mask.copy()
        if open_px >= 2:
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
        if close_px >= 2:
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
        return out

    def keep_large_blobs(self, mask: np.ndarray):
        min_area = int(self.get_parameter("min_free_blob_area_px").value)
        if min_area <= 0:
            return mask
        num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        out = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    def vertical_depth_fallback(self, depth_roi: np.ndarray):
        zmin = float(self.get_parameter("min_valid_depth_m").value)
        zmax = self.depth_limit()
        valid = np.isfinite(depth_roi) & (depth_roi > zmin) & (depth_roi < zmax)

        dz_up = np.full_like(depth_roi, np.nan, dtype=np.float32)
        dz_up[1:, :] = depth_roi[:-1, :] - depth_roi[1:, :]

        same_eps = float(self.get_parameter("vertical_same_depth_eps_m").value)
        large_jump = float(self.get_parameter("vertical_large_jump_m").value)

        not_wall = np.isfinite(dz_up) & (np.abs(dz_up) > same_eps) & (np.abs(dz_up) < large_jump)
        free = valid & not_wall

        bottom_rows = max(8, int(0.10 * depth_roi.shape[0]))
        free[-bottom_rows:, :] = valid[-bottom_rows:, :]

        free = self.clean_mask((free.astype(np.uint8) * 255))
        free = self.keep_large_blobs(free)
        return free, {"plane_n": None, "plane_d": None}

    def make_overlay(self, frame: np.ndarray, free_roi: np.ndarray, box, aux):
        out = frame.copy()
        x0, y0, x1, y1 = box
        roi = out[y0:y1, x0:x1]

        free = free_roi > 0
        color = np.zeros_like(roi)
        color[free] = (0, 185, 0)
        color[~free] = (0, 0, 185)

        blended = cv2.addWeighted(roi, 0.55, color, 0.45, 0)
        out[y0:y1, x0:x1] = blended
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 2)

        if aux is not None and aux.get("obstacle") is not None:
            obs = aux["obstacle"] > 0
            edge = cv2.Canny((obs.astype(np.uint8) * 255), 50, 150)
            contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            shifted = []
            for c in contours:
                c = c.copy()
                c[:, 0, 0] += x0
                c[:, 0, 1] += y0
                shifted.append(c)
            cv2.drawContours(out, shifted, -1, (0, 255, 255), 1)

        mode = str(self.get_parameter("depth_limit_mode").value)
        tag = "none" if self.tag_distance_m is None else f"{self.tag_distance_m:.2f}m"
        cv2.putText(out, "Depth freespace: green=drivable, red=not drivable",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 2)
        cv2.putText(out, f"depth_limit={self.depth_limit():.2f}m mode={mode} tag={tag}",
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2)

        if aux is not None and aux.get("plane_n") is not None:
            n = aux["plane_n"]
            cv2.putText(out, f"ground plane n=[{n[0]:.2f},{n[1]:.2f},{n[2]:.2f}]",
                        (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        else:
            cv2.putText(out, "ground plane fallback mode",
                        (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        return out

    def make_topdown(self, depth_roi: np.ndarray, free_roi: np.ndarray, box):
        projected = self.project_roi_to_points(depth_roi, box)
        width_m = float(self.get_parameter("topdown_width_m").value)
        forward_m = float(self.get_parameter("topdown_forward_m").value)
        res = float(self.get_parameter("topdown_resolution_m").value)

        nx = int(forward_m / res)
        ny = int(width_m / res)
        img = np.zeros((nx, ny, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)

        if projected is None:
            return img

        pts, valid_small, uu, vv = projected
        x_lat = pts[:, :, 0]
        z_fwd = pts[:, :, 2]

        free_small = free_roi[vv, uu] > 0
        keep = valid_small & (z_fwd > 0) & (z_fwd < forward_m) & (np.abs(x_lat) < width_m / 2)

        ix = np.floor(z_fwd[keep] / res).astype(np.int32)
        iy = np.floor((x_lat[keep] + width_m / 2) / res).astype(np.int32)
        free_vals = free_small[keep]

        img[ix[free_vals], iy[free_vals]] = (0, 180, 0)
        img[ix[~free_vals], iy[~free_vals]] = (0, 0, 180)

        img = cv2.flip(img, 0)
        img = cv2.resize(img, (280, 420), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, "top-down", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(img, "front", (105, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        return img

    def tick(self):
        if self.rgb is None or self.depth_m is None:
            return

        frame = self.rgb.copy()
        box = self.crop_box(frame.shape)
        x0, y0, x1, y1 = box

        depth_roi = self.depth_m[y0:y1, x0:x1]
        if depth_roi.size == 0:
            return

        free_roi, aux = self.classify_freespace(depth_roi, box)
        if free_roi is None:
            return

        overlay = self.make_overlay(frame, free_roi, box, aux)

        if bool(self.get_parameter("show_topdown").value):
            topdown = self.make_topdown(depth_roi, free_roi, box)
            topdown_h = overlay.shape[0]
            scale = topdown_h / topdown.shape[0]
            topdown = cv2.resize(topdown, (int(topdown.shape[1] * scale), topdown_h))
            view = np.hstack([overlay, topdown])
        else:
            view = overlay

        cv2.imshow("depth_freespace_viewer", view)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = DepthFreespaceViewer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
