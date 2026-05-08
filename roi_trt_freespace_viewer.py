#!/usr/bin/env python3
"""
roi_trt_freespace_viewer.py

ROS 2 viewer for cropped ROI segmentation from Kinect RGB/depth.

Purpose:
- Run a TensorRT segmentation engine on a cropped lower-image ROI.
- Show whether the model marks free drive zone vs obstacles correctly.
- Optionally mask out pixels farther than the AprilTag, because those do not matter
  for driving toward the tag.
- Overlay segmentation on the RGB frame so you can see what it sees.

No PyTorch is used at runtime.

Expected model:
- TensorRT engine with one image input, usually NCHW RGB float32/float16.
- Output can be:
    [1, 1, H, W] probability/logit for free space
    [1, 2, H, W] logits/probs where channel 0/1 are obstacle/free, configurable
    [1, H, W] probability/logit for free space

If you do not have a TensorRT engine yet, run with:
  --ros-args -p use_depth_fallback:=true
This does NOT use neural segmentation; it just gives a quick depth-based free/obstacle overlay for debugging.
"""

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


try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TRT_AVAILABLE = True
except Exception:
    TRT_AVAILABLE = False


class TensorRTSegmenter:
    def __init__(self, engine_path: str, free_channel: int = 1):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT/pycuda import failed. Install TensorRT Python bindings and pycuda.")
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Could not load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.free_channel = int(free_channel)

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Engine must have one input and one output tensor.")

        in_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        # For dynamic shapes, require parameters to set context before allocation.
        if any(d < 0 for d in in_shape):
            raise RuntimeError(
                "Dynamic input TensorRT engine detected. Rebuild with fixed input size, "
                "or extend this script to set_binding_shape/context tensor shape."
            )

        self.input_shape = in_shape  # normally [1,3,H,W]
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        self.h_input = cuda.pagelocked_empty(int(np.prod(self.input_shape)), self.input_dtype)
        self.h_output = cuda.pagelocked_empty(int(np.prod(self.output_shape)), self.output_dtype)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

    def infer(self, bgr_roi: np.ndarray) -> np.ndarray:
        n, c, h, w = self.input_shape
        if n != 1 or c not in (1, 3):
            raise RuntimeError(f"Expected input shape [1,3,H,W] or [1,1,H,W], got {self.input_shape}")

        resized = cv2.resize(bgr_roi, (w, h), interpolation=cv2.INTER_LINEAR)
        if c == 3:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            x = rgb.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))[None, ...]
        else:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            x = gray[None, None, ...]

        x = x.astype(self.input_dtype, copy=False)
        np.copyto(self.h_input, x.ravel())

        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        y = np.array(self.h_output, dtype=np.float32).reshape(self.output_shape)
        mask_small = self.output_to_free_mask(y)
        mask = cv2.resize(mask_small.astype(np.float32), (bgr_roi.shape[1], bgr_roi.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        return mask

    def output_to_free_mask(self, y: np.ndarray) -> np.ndarray:
        # y possibilities:
        # [1, 1, H, W], [1, 2, H, W], [1, H, W], [H, W]
        y = np.squeeze(y)
        if y.ndim == 3:
            # C,H,W
            if y.shape[0] == 1:
                z = y[0]
                return self.sigmoid_or_prob(z)
            else:
                ch = min(max(self.free_channel, 0), y.shape[0] - 1)
                # Softmax over channels if not already probabilities.
                yy = y - np.max(y, axis=0, keepdims=True)
                exp = np.exp(yy)
                prob = exp / np.maximum(np.sum(exp, axis=0, keepdims=True), 1e-6)
                return prob[ch]
        if y.ndim == 2:
            return self.sigmoid_or_prob(y)
        raise RuntimeError(f"Unsupported output shape after squeeze: {y.shape}")

    @staticmethod
    def sigmoid_or_prob(z: np.ndarray) -> np.ndarray:
        # If output already looks like 0..1 probability, keep it.
        if np.nanmin(z) >= 0.0 and np.nanmax(z) <= 1.0:
            return z.astype(np.float32)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


class ROIFreespaceViewer(Node):
    def __init__(self):
        super().__init__("roi_trt_freespace_viewer")
        self.bridge = CvBridge()

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("tag_pose_topic", "/tag_pose")
        self.declare_parameter("tag_distance_topic", "/tag_distance")

        self.declare_parameter("engine_path", "")
        self.declare_parameter("free_channel", 1)
        self.declare_parameter("free_threshold", 0.50)

        # Crop lower part of the RGB image; this is where free drive zone usually appears.
        self.declare_parameter("roi_x_min_frac", 0.05)
        self.declare_parameter("roi_x_max_frac", 0.95)
        self.declare_parameter("roi_y_min_frac", 0.35)
        self.declare_parameter("roi_y_max_frac", 1.00)

        # Slower segmentation rate than camera rate.
        self.declare_parameter("segmentation_hz", 7.5)

        # Ignore pixels farther than the tag.
        self.declare_parameter("use_tag_depth_gate", True)
        self.declare_parameter("tag_depth_buffer_m", 0.08)
        self.declare_parameter("max_depth_without_tag_m", 2.0)

        # Optional debug fallback: no neural model, just depth threshold.
        self.declare_parameter("use_depth_fallback", False)
        self.declare_parameter("fallback_obstacle_near_m", 0.15)

        self.rgb: Optional[np.ndarray] = None
        self.depth_m: Optional[np.ndarray] = None
        self.tag_distance_m: Optional[float] = None
        self.last_mask_full: Optional[np.ndarray] = None

        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        info_topic = self.get_parameter("camera_info_topic").value
        tag_pose_topic = self.get_parameter("tag_pose_topic").value
        tag_distance_topic = self.get_parameter("tag_distance_topic").value

        self.create_subscription(Image, rgb_topic, self.rgb_cb, 5)
        self.create_subscription(Image, depth_topic, self.depth_cb, 5)
        self.create_subscription(CameraInfo, info_topic, self.info_cb, 5)
        self.create_subscription(PoseStamped, tag_pose_topic, self.tag_pose_cb, 5)
        self.create_subscription(Float32, tag_distance_topic, self.tag_distance_cb, 5)

        self.segmenter: Optional[TensorRTSegmenter] = None
        engine_path = str(self.get_parameter("engine_path").value)
        use_fallback = bool(self.get_parameter("use_depth_fallback").value)

        if engine_path:
            self.segmenter = TensorRTSegmenter(
                engine_path=engine_path,
                free_channel=int(self.get_parameter("free_channel").value),
            )
            self.get_logger().info(f"Loaded TensorRT segmentation engine: {engine_path}")
        elif not use_fallback:
            self.get_logger().warn(
                "No engine_path provided. Set engine_path to a TensorRT .engine file, "
                "or set use_depth_fallback:=true for a non-neural debug overlay."
            )

        hz = float(self.get_parameter("segmentation_hz").value)
        self.timer = self.create_timer(1.0 / max(hz, 0.1), self.tick)
        self.get_logger().info("ROIFreespaceViewer ready")

    def rgb_cb(self, msg: Image):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def depth_cb(self, msg: Image):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if d.dtype == np.uint16:
            self.depth_m = d.astype(np.float32) / 1000.0
        else:
            self.depth_m = d.astype(np.float32)

    def info_cb(self, msg: CameraInfo):
        pass

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

    def depth_gate_mask(self, depth_roi: np.ndarray) -> np.ndarray:
        max_no_tag = float(self.get_parameter("max_depth_without_tag_m").value)
        tag_buf = float(self.get_parameter("tag_depth_buffer_m").value)
        if bool(self.get_parameter("use_tag_depth_gate").value) and self.tag_distance_m is not None:
            max_d = max(0.1, min(max_no_tag, self.tag_distance_m - tag_buf))
        else:
            max_d = max_no_tag
        return np.isfinite(depth_roi) & (depth_roi > 0.05) & (depth_roi < max_d)

    def fallback_segment(self, depth_roi: np.ndarray) -> np.ndarray:
        """
        Crude non-neural debug mask:
        free = valid depth up to tag and not extremely close.
        This is not semantic segmentation; it is only a viewer sanity check.
        """
        valid = self.depth_gate_mask(depth_roi)
        near = float(self.get_parameter("fallback_obstacle_near_m").value)
        free = valid & (depth_roi > near)
        # Clean it up visually.
        free_u8 = (free.astype(np.uint8) * 255)
        free_u8 = cv2.morphologyEx(free_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        free_u8 = cv2.morphologyEx(free_u8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return free_u8.astype(np.float32) / 255.0

    def overlay(self, frame: np.ndarray, free_mask_full: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        out = frame.copy()
        x0, y0, x1, y1 = box
        threshold = float(self.get_parameter("free_threshold").value)

        roi = out[y0:y1, x0:x1]
        mask = free_mask_full[y0:y1, x0:x1]
        free = mask >= threshold
        obstacle = ~free

        color = np.zeros_like(roi)
        color[free] = (0, 180, 0)       # free drive zone
        color[obstacle] = (0, 0, 180)   # obstacle / not-free

        out[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.55, color, 0.45, 0.0)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 2)

        tag_txt = f"tag depth gate: {self.tag_distance_m:.2f}m" if self.tag_distance_m is not None else "tag depth gate: no tag"
        mode = "TensorRT" if self.segmenter is not None else "depth fallback"
        cv2.putText(out, mode, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.putText(out, tag_txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 0), 2)
        cv2.putText(out, "green=free, red=obstacle/not-free", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        return out

    def tick(self):
        if self.rgb is None:
            return
        frame = self.rgb.copy()
        h, w = frame.shape[:2]
        box = self.crop_box(frame.shape)
        x0, y0, x1, y1 = box
        rgb_roi = frame[y0:y1, x0:x1]

        free_mask_roi = None
        if self.segmenter is not None:
            free_mask_roi = self.segmenter.infer(rgb_roi)
        elif bool(self.get_parameter("use_depth_fallback").value):
            if self.depth_m is None:
                return
            depth_roi = self.depth_m[y0:y1, x0:x1]
            free_mask_roi = self.fallback_segment(depth_roi)
        else:
            cv2.imshow("roi_trt_freespace_viewer", frame)
            cv2.waitKey(1)
            return

        # Apply the "closer than tag" gate to neural segmentation too.
        if self.depth_m is not None:
            depth_roi = self.depth_m[y0:y1, x0:x1]
            gate = self.depth_gate_mask(depth_roi)
            free_mask_roi = free_mask_roi * gate.astype(np.float32)

        free_mask_full = np.zeros((h, w), dtype=np.float32)
        free_mask_full[y0:y1, x0:x1] = free_mask_roi
        vis = self.overlay(frame, free_mask_full, box)

        cv2.imshow("roi_trt_freespace_viewer: free drive zone segmentation", vis)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ROIFreespaceViewer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
