"""
双目2D关键点录制程序 + AprilTag检测
只记录左右摄像头的2D关键点坐标，不计算3D坐标
输出文件：
- keypoints_2d_left.csv 和 keypoints_2d_right.csv (关键点数据)
- apriltag_left.csv 和 apriltag_right.csv (AprilTag位姿数据)
"""
import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
import yaml

from runtime_compat import (
    CAMERA_BACKEND_CHOICES,
    DEFAULT_LEFT_CAMERA_ID,
    DEFAULT_RIGHT_CAMERA_ID,
    open_stereo_captures,
    resolve_inference_device,
)

try:
    from pupil_apriltags import Detector as AprilTagDetector
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False
    print("警告: pupil_apriltags 未安装，AprilTag检测功能将被禁用")
    print("安装方法: pip install pupil-apriltags")


def load_camera_params_from_yaml(yaml_path):
    """从YAML配置文件加载相机内参
    
    返回: (camera_params_left, camera_params_right)
    每个参数为 [fx, fy, cx, cy] 格式
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 提取左相机内参
        cam_mat_left = config.get('camera_matrix_left', {}).get('data', [])
        if len(cam_mat_left) >= 9:
            # 相机矩阵格式: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            params_left = [cam_mat_left[0], cam_mat_left[4], cam_mat_left[2], cam_mat_left[5]]
        else:
            params_left = None
        
        # 提取右相机内参
        cam_mat_right = config.get('camera_matrix_right', {}).get('data', [])
        if len(cam_mat_right) >= 9:
            params_right = [cam_mat_right[0], cam_mat_right[4], cam_mat_right[2], cam_mat_right[5]]
        else:
            params_right = None
        
        return params_left, params_right
    except Exception as e:
        print(f"警告: 无法从YAML文件加载相机内参: {e}")
        return None, None


def _set_capture_properties(cap: cv2.VideoCapture, width: int, height: int, fps: int) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    # 尽量减少相机缓冲堆积（不保证所有后端生效，但值得一试）
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


# Halpe26（常用顺序）下半身点：髋/膝/踝 + 大脚趾/脚跟
# left:  11 13 15 19 21
# right: 12 14 16 22 24
# 添加19(hip中心点)，但不在连线中
LOWER_KPT_IDXS = [11, 12, 13, 14, 15, 16, 19, 20, 21]
KPT_NAME = {
    11: "lhip",
    12: "rhip",
    13: "lkne",
    14: "rkne",
    15: "lank",
    16: "rank",
    19: "hip",
    20: "ltoe",
    21: "rtoe",
}

# 简单下肢骨架连线（按需可加）- 不连接hip点
LOWER_EDGES = [
    (11, 13), (13, 15),  # left hip-knee-ankle 左腿
    (12, 14), (14, 16),  # right hip-knee-ankle 右腿
    (15, 20),            # 左脚踝连脚趾
    (16, 21)             # 右脚踝连脚趾
]


def _draw_lower_only(
    img: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    kpt_thr: float,
    show_names: bool = True,
) -> np.ndarray:
    """只画下半身指定关键点与连线(OpenCV画,不用mmpose visualizer)。
    优化版本：更小的关节点，更细的绿色连杆，显示所有hip点但不连接。
    """
    # 先画连线（在点下面）
    for a, b in LOWER_EDGES:
        ca = float(kpts_conf[a])
        cb = float(kpts_conf[b])
        if ca < kpt_thr or cb < kpt_thr:
            continue
        x1, y1 = kpts_xy[a]
        x2, y2 = kpts_xy[b]
        xi1, yi1 = int(round(float(x1))), int(round(float(y1)))
        xi2, yi2 = int(round(float(x2))), int(round(float(y2)))
        
        # 边界检查
        if not (0 <= xi1 < img.shape[1] and 0 <= yi1 < img.shape[0]):
            continue
        if not (0 <= xi2 < img.shape[1] and 0 <= yi2 < img.shape[0]):
            continue
        
        # 绿色，1像素粗细
        cv2.line(img, (xi1, yi1), (xi2, yi2), (0, 255, 0), 1, lineType=cv2.LINE_AA)

    # 再画关节点（在连线上面）
    for idx in LOWER_KPT_IDXS:
        c = float(kpts_conf[idx])
        if c < kpt_thr:
            continue
        x, y = kpts_xy[idx]
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if not (0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]):
            continue

        # 更小的关节点：主要关节3像素，其他2像素
        r = 3 if idx in (11, 12, 13, 14, 15, 16, 19) else 2
        # 绿色关节点
        cv2.circle(img, (xi, yi), r, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        if show_names:
            name = KPT_NAME.get(idx, str(idx))
            # 绿色文字
            cv2.putText(
                img,
                name,
                (min(xi + 5, img.shape[1] - 1), max(yi - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,  # 稍小的字体
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    return img


class PoseRunner:
    def __init__(
        self,
        det_config: str,
        det_checkpoint: str,
        pose_config: str,
        pose_checkpoint: str,
        device: str,
        det_cat_id: int = 0,
        bbox_thr: float = 0.3,
        nms_thr: float = 0.3,
        kpt_thr: float = 0.3,
        show_names: bool = True,
    ) -> None:
        try:
            import torch
            from mmdet.apis import inference_detector, init_detector
            from mmpose.apis import inference_topdown
            from mmpose.apis import init_model as init_pose_estimator
            from mmpose.evaluation.functional import nms
            from mmpose.structures import merge_data_samples
            from mmpose.utils import adapt_mmdet_pipeline
        except Exception as e:
            raise RuntimeError(
                "Failed to import mmdet/mmpose dependencies. "
                "Please ensure mmdet and mmpose are installed in this environment."
            ) from e

        device = resolve_inference_device(device, torch)

        self._torch = torch
        self._inference_detector = inference_detector
        self._inference_topdown = inference_topdown
        self._nms = nms
        self._merge_data_samples = merge_data_samples
        self._det_cat_id = det_cat_id
        self._bbox_thr = bbox_thr
        self._nms_thr = nms_thr
        self._kpt_thr = kpt_thr
        self._show_names = show_names

        self.detector = init_detector(det_config, det_checkpoint, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        self.pose_estimator = init_pose_estimator(
            pose_config,
            pose_checkpoint,
            device=device,
        )

    def _process_detection(self, frame_bgr: np.ndarray):
        """优化：提取公共检测逻辑"""
        det_result = self._inference_detector(self.detector, frame_bgr)
        pred_instance = det_result.pred_instances.cpu().numpy()

        bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[
            np.logical_and(pred_instance.labels == self._det_cat_id, pred_instance.scores > self._bbox_thr)
        ]
        if bboxes.shape[0] == 0:
            return None

        keep = self._nms(bboxes, self._nms_thr)
        bboxes = bboxes[keep, :]
        if bboxes.shape[0] == 0:
            return None

        # 只取主目标 bbox（面积*score 最大）
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        scores = bboxes[:, 4]
        best = int(np.argmax(areas * scores))
        return bboxes[best:best + 1, :4]

    def visualize_and_get_lower_kpts(self, frame_bgr: np.ndarray):
        """优化：合并可视化和关键点提取，减少重复计算"""
        with self._torch.no_grad():
            bboxes = self._process_detection(frame_bgr)
            if bboxes is None:
                return frame_bgr, None

            pose_results = self._inference_topdown(self.pose_estimator, frame_bgr, bboxes)
            data_samples = self._merge_data_samples(pose_results)

        pred = data_samples.get("pred_instances", None)
        if pred is None:
            return frame_bgr, None

        kpts = getattr(pred, "keypoints", None)
        kpt_scores = getattr(pred, "keypoint_scores", None)
        if kpts is None:
            return frame_bgr, None

        if not isinstance(kpts, np.ndarray):
            kpts = kpts.detach().cpu().numpy()
        k0 = kpts[0]

        if kpt_scores is None:
            c0 = np.ones((k0.shape[0],), dtype=np.float32)
        else:
            if not isinstance(kpt_scores, np.ndarray):
                kpt_scores = kpt_scores.detach().cpu().numpy()
            c0 = kpt_scores[0]

        # 直接在原图上绘制，避免copy
        _draw_lower_only(frame_bgr, k0, c0, self._kpt_thr, show_names=self._show_names)

        # 提取下半身关键点
        out = {}
        for idx in LOWER_KPT_IDXS:
            x, y = float(k0[idx][0]), float(k0[idx][1])
            conf = float(c0[idx])
            out[idx] = (x, y, conf)
        
        return frame_bgr, out


def run_stereo_capture(
    left_source: int | str = DEFAULT_LEFT_CAMERA_ID,
    right_source: int | str = DEFAULT_RIGHT_CAMERA_ID,
    width: int = 1920,
    height: int = 1080,
    fps: int = 60,
    display_mode: str = "stitch",
    backend: str = "auto",
    pose_runner: PoseRunner | None = None,
    pose_side: str = "left",
    infer_interval: int = 1,
    kpt_thr: float = 0.3,
    save_2d_csv: bool = True,
    warmup_s: float = 0.0,
    drop_frames: int = 0,
    use_grab: bool = False,
    start_on_key: bool = False,
    start_key: str = "s",
    save_video: str = "",
    enable_apriltag: bool = False,
    apriltag_family: str = "tag36h11",
    apriltag_size: float = 0.05,
    camera_params_left: list = None,
    camera_params_right: list = None,
) -> None:
    left_cap, right_cap = open_stereo_captures(
        left_source, right_source, backend=backend)

    _set_capture_properties(left_cap, width, height, fps)
    _set_capture_properties(right_cap, width, height, fps)

    use_grab_mode = bool(use_grab)

    if warmup_s > 0:
        t_end = time.perf_counter() + float(warmup_s)
        while time.perf_counter() < t_end:
            if use_grab_mode:
                left_cap.grab()
                right_cap.grab()
            else:
                left_cap.read()
                right_cap.read()

    drop_n = int(drop_frames)
    if drop_n > 0:
        for _ in range(drop_n):
            if use_grab_mode:
                left_cap.grab()
                right_cap.grab()
            else:
                left_cap.read()
                right_cap.read()

    if display_mode not in {"separate", "stitch"}:
        raise ValueError("display_mode must be 'separate' or 'stitch'")

    if display_mode == "separate":
        left_window = "Left"
        right_window = "Right"
        cv2.namedWindow(left_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(right_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(left_window, width, height)
        cv2.resizeWindow(right_window, width, height)
    else:
        stitched_window = "Stereo (L|R)"
        cv2.namedWindow(stitched_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(stitched_window, width * 2, height)

    last_t = time.perf_counter()
    fps_ema = 0.0
    alpha = 0.1
    frame_idx = 0
    last_kpts_l = None
    last_kpts_r = None
    # 保存上一帧的可视化结果，避免频闪
    last_vis_l = None
    last_vis_r = None

    # 创建recordings文件夹和时间戳子文件夹
    recordings_dir = "recordings"
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(recordings_dir, timestamp)
    
    csv_f_left = None
    csv_f_right = None
    csv_writer_left = None
    csv_writer_right = None
    csv_f_tag_left = None
    csv_f_tag_right = None
    csv_writer_tag_left = None
    csv_writer_tag_right = None
    pending_session_dir = ""
    
    # 初始化AprilTag检测器
    apriltag_detector = None
    if enable_apriltag and APRILTAG_AVAILABLE:
        try:
            apriltag_detector = AprilTagDetector(
                families=apriltag_family,
                nthreads=2,
                quad_decimate=2.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0
            )
            print(f"AprilTag检测器已初始化: {apriltag_family}, 标签尺寸: {apriltag_size}m")
        except Exception as e:
            print(f"警告: AprilTag检测器初始化失败: {e}")
            apriltag_detector = None
    
    if save_2d_csv:
        if start_on_key:
            pending_session_dir = session_dir
        else:
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)
            # 创建左右两个CSV文件（关键点）
            csv_path_left = os.path.join(session_dir, "keypoints_2d_left.csv")
            csv_path_right = os.path.join(session_dir, "keypoints_2d_right.csv")
            csv_f_left = open(csv_path_left, "w", newline="", encoding="utf-8")
            csv_f_right = open(csv_path_right, "w", newline="", encoding="utf-8")
            csv_writer_left = csv.writer(csv_f_left)
            csv_writer_right = csv.writer(csv_f_right)
            # CSV表头
            csv_writer_left.writerow(["frame", "kpt_idx", "kpt_name", "x", "y", "confidence"])
            csv_writer_right.writerow(["frame", "kpt_idx", "kpt_name", "x", "y", "confidence"])
            
            # 创建AprilTag CSV文件
            if apriltag_detector is not None:
                csv_path_tag_left = os.path.join(session_dir, "apriltag_left.csv")
                csv_path_tag_right = os.path.join(session_dir, "apriltag_right.csv")
                csv_f_tag_left = open(csv_path_tag_left, "w", newline="", encoding="utf-8")
                csv_f_tag_right = open(csv_path_tag_right, "w", newline="", encoding="utf-8")
                csv_writer_tag_left = csv.writer(csv_f_tag_left)
                csv_writer_tag_right = csv.writer(csv_f_tag_right)
                # AprilTag CSV表头
                header = ["frame", "tag_id", "decision_margin", "hamming", 
                         "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33",
                         "tx", "ty", "tz", "pose_error"]
                csv_writer_tag_left.writerow(header)
                csv_writer_tag_right.writerow(header)

    recording = (not start_on_key)
    record_frame_idx = 0
    start_key_norm = (str(start_key).strip().lower()[:1] if start_key else "s")

    # 视频写入器：原始视频和识别后的视频
    vw_raw_l = None
    vw_raw_r = None
    vw_raw_stitch = None
    vw_vis_l = None
    vw_vis_r = None
    vw_vis_stitch = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    # 计算实际录制帧率（考虑推理间隔）
    actual_record_fps = float(fps) / float(infer_interval) if infer_interval > 0 else float(fps)
    print(f"相机FPS: {fps}, 推理间隔: {infer_interval}, 实际录制FPS: {actual_record_fps:.2f}")

    grab_fallback_to_read = False

    try:
        while True:
            frame_idx += 1
            if use_grab_mode and (not grab_fallback_to_read):
                ok_gl = left_cap.grab()
                ok_gr = right_cap.grab()
                if ok_gl and ok_gr:
                    ok_l, frame_l = left_cap.retrieve()
                    ok_r, frame_r = right_cap.retrieve()
                else:
                    grab_fallback_to_read = True
                    ok_l, frame_l = left_cap.read()
                    ok_r, frame_r = right_cap.read()
            else:
                ok_l, frame_l = left_cap.read()
                ok_r, frame_r = right_cap.read()

            if not ok_l or frame_l is None:
                print("Left camera read failed. Exiting...")
                break
            if not ok_r or frame_r is None:
                print("Right camera read failed. Exiting...")
                break

            # 优化：只在需要时resize
            if frame_l.shape[1] != width or frame_l.shape[0] != height:
                frame_l = cv2.resize(frame_l, (width, height), interpolation=cv2.INTER_LINEAR)
            if frame_r.shape[1] != width or frame_r.shape[0] != height:
                frame_r = cv2.resize(frame_r, (width, height), interpolation=cv2.INTER_LINEAR)

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            inst_fps = (1.0 / dt) if dt > 0 else 0.0
            fps_ema = inst_fps if fps_ema <= 0 else (fps_ema * (1.0 - alpha) + inst_fps * alpha)

            do_infer = pose_runner is not None and infer_interval > 0 and (frame_idx % infer_interval == 0)
            
            # 优化：保存原始帧的引用（用于录制）
            frame_l_raw = frame_l
            frame_r_raw = frame_r
            
            if do_infer:
                if pose_side == "both":
                    # 优化：复制帧用于可视化，避免修改原始帧
                    frame_l_vis = frame_l.copy()
                    frame_r_vis = frame_r.copy()
                    frame_l_vis, last_kpts_l = pose_runner.visualize_and_get_lower_kpts(frame_l_vis)
                    frame_r_vis, last_kpts_r = pose_runner.visualize_and_get_lower_kpts(frame_r_vis)
                    # 保存这一帧的可视化结果
                    last_vis_l = frame_l_vis
                    last_vis_r = frame_r_vis
                else:
                    frame_l_vis = frame_l.copy() if pose_side in {"left", "both"} else frame_l
                    frame_r_vis = frame_r.copy() if pose_side in {"right", "both"} else frame_r
                    last_kpts_l = None
                    last_kpts_r = None
                    last_vis_l = frame_l_vis
                    last_vis_r = frame_r_vis
            else:
                # 在当前帧上重新绘制上一次检测到的关键点，避免频闪
                frame_l_vis = frame_l.copy()
                frame_r_vis = frame_r.copy()
                
                # 如果有上一次的关键点数据，在新帧上重新绘制
                if last_kpts_l is not None and pose_side in {"left", "both"}:
                    # 先绘制连线
                    for a, b in LOWER_EDGES:
                        if a in last_kpts_l and b in last_kpts_l:
                            xa, ya, ca = last_kpts_l[a]
                            xb, yb, cb = last_kpts_l[b]
                            if ca >= kpt_thr and cb >= kpt_thr:
                                p1 = (int(round(xa)), int(round(ya)))
                                p2 = (int(round(xb)), int(round(yb)))
                                if (0 <= p1[0] < width and 0 <= p1[1] < height and 
                                    0 <= p2[0] < width and 0 <= p2[1] < height):
                                    cv2.line(frame_l_vis, p1, p2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                    # 再绘制关键点和标签
                    for idx in LOWER_KPT_IDXS:
                        if idx in last_kpts_l:
                            x, y, conf = last_kpts_l[idx]
                            if conf >= kpt_thr:
                                xi, yi = int(round(x)), int(round(y))
                                if 0 <= xi < width and 0 <= yi < height:
                                    r = 3 if idx in (11, 12, 13, 14, 15, 16, 19) else 2
                                    cv2.circle(frame_l_vis, (xi, yi), r, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                                    # 绘制标签
                                    if pose_runner._show_names:
                                        name = KPT_NAME.get(idx, str(idx))
                                        cv2.putText(
                                            frame_l_vis,
                                            name,
                                            (min(xi + 5, width - 1), max(yi - 5, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.4,
                                            (0, 255, 0),
                                            1,
                                            cv2.LINE_AA,
                                        )
                else:
                    frame_l_vis = frame_l
                
                if last_kpts_r is not None and pose_side in {"right", "both"}:
                    # 先绘制连线
                    for a, b in LOWER_EDGES:
                        if a in last_kpts_r and b in last_kpts_r:
                            xa, ya, ca = last_kpts_r[a]
                            xb, yb, cb = last_kpts_r[b]
                            if ca >= kpt_thr and cb >= kpt_thr:
                                p1 = (int(round(xa)), int(round(ya)))
                                p2 = (int(round(xb)), int(round(yb)))
                                if (0 <= p1[0] < width and 0 <= p2[0] < width and 
                                    0 <= p1[1] < height and 0 <= p2[1] < height):
                                    cv2.line(frame_r_vis, p1, p2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                    # 再绘制关键点和标签
                    for idx in LOWER_KPT_IDXS:
                        if idx in last_kpts_r:
                            x, y, conf = last_kpts_r[idx]
                            if conf >= kpt_thr:
                                xi, yi = int(round(x)), int(round(y))
                                if 0 <= xi < width and 0 <= yi < height:
                                    r = 3 if idx in (11, 12, 13, 14, 15, 16, 19) else 2
                                    cv2.circle(frame_r_vis, (xi, yi), r, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                                    # 绘制标签
                                    if pose_runner._show_names:
                                        name = KPT_NAME.get(idx, str(idx))
                                        cv2.putText(
                                            frame_r_vis,
                                            name,
                                            (min(xi + 5, width - 1), max(yi - 5, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.4,
                                            (0, 255, 0),
                                            1,
                                            cv2.LINE_AA,
                                        )
                else:
                    frame_r_vis = frame_r

            allow_record = bool(recording)
            if allow_record:
                record_frame_idx += 1
            frame_out_idx = int(record_frame_idx) if start_on_key else int(frame_idx)

            # AprilTag检测
            tags_left = []
            tags_right = []
            if apriltag_detector is not None:
                # 转换为灰度图并进行直方图均衡化
                gray_l = cv2.cvtColor(frame_l_raw, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r_raw, cv2.COLOR_BGR2GRAY)
                gray_l_eq = cv2.equalizeHist(gray_l)
                gray_r_eq = cv2.equalizeHist(gray_r)
                
                # 检测AprilTag
                if camera_params_left is not None:
                    tags_left = apriltag_detector.detect(
                        gray_l_eq, 
                        estimate_tag_pose=True,
                        camera_params=camera_params_left,
                        tag_size=apriltag_size
                    )
                else:
                    tags_left = apriltag_detector.detect(gray_l_eq, estimate_tag_pose=False)
                
                if camera_params_right is not None:
                    tags_right = apriltag_detector.detect(
                        gray_r_eq,
                        estimate_tag_pose=True,
                        camera_params=camera_params_right,
                        tag_size=apriltag_size
                    )
                else:
                    tags_right = apriltag_detector.detect(gray_r_eq, estimate_tag_pose=False)
                
                # 在可视化图像上绘制AprilTag
                for tag in tags_left:
                    # 绘制边界
                    for idx in range(len(tag.corners)):
                        cv2.line(frame_l_vis,
                                tuple(tag.corners[idx-1, :].astype(int)),
                                tuple(tag.corners[idx, :].astype(int)),
                                (255, 0, 255), 2)
                    center = tuple(tag.center.astype(int))
                    cv2.circle(frame_l_vis, center, 5, (255, 0, 255), -1)
                    cv2.putText(frame_l_vis, f"Tag{tag.tag_id}", 
                               (center[0]-20, center[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                for tag in tags_right:
                    for idx in range(len(tag.corners)):
                        cv2.line(frame_r_vis,
                                tuple(tag.corners[idx-1, :].astype(int)),
                                tuple(tag.corners[idx, :].astype(int)),
                                (255, 0, 255), 2)
                    center = tuple(tag.center.astype(int))
                    cv2.circle(frame_r_vis, center, 5, (255, 0, 255), -1)
                    cv2.putText(frame_r_vis, f"Tag{tag.tag_id}",
                               (center[0]-20, center[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # 保存2D关键点数据到CSV
            if allow_record and save_2d_csv:
                # 保存左相机2D数据
                if csv_writer_left is not None and last_kpts_l is not None:
                    for idx in LOWER_KPT_IDXS:
                        if idx in last_kpts_l:
                            x, y, conf = last_kpts_l[idx]
                            csv_writer_left.writerow([
                                frame_out_idx, idx, KPT_NAME.get(idx, str(idx)),
                                float(x), float(y), float(conf)
                            ])
                
                # 保存右相机2D数据
                if csv_writer_right is not None and last_kpts_r is not None:
                    for idx in LOWER_KPT_IDXS:
                        if idx in last_kpts_r:
                            x, y, conf = last_kpts_r[idx]
                            csv_writer_right.writerow([
                                frame_out_idx, idx, KPT_NAME.get(idx, str(idx)),
                                float(x), float(y), float(conf)
                            ])
                
                # 保存AprilTag数据
                if csv_writer_tag_left is not None and len(tags_left) > 0:
                    for tag in tags_left:
                        if hasattr(tag, 'pose_R') and tag.pose_R is not None:
                            R = tag.pose_R.flatten()
                            t = tag.pose_t.flatten()
                            pose_err = tag.pose_err if hasattr(tag, 'pose_err') else 0.0
                            csv_writer_tag_left.writerow([
                                frame_out_idx, tag.tag_id, tag.decision_margin, tag.hamming,
                                R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
                                t[0], t[1], t[2], pose_err
                            ])
                        else:
                            # 没有位姿估计，只保存检测信息
                            csv_writer_tag_left.writerow([
                                frame_out_idx, tag.tag_id, tag.decision_margin, tag.hamming,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            ])
                
                if csv_writer_tag_right is not None and len(tags_right) > 0:
                    for tag in tags_right:
                        if hasattr(tag, 'pose_R') and tag.pose_R is not None:
                            R = tag.pose_R.flatten()
                            t = tag.pose_t.flatten()
                            pose_err = tag.pose_err if hasattr(tag, 'pose_err') else 0.0
                            csv_writer_tag_right.writerow([
                                frame_out_idx, tag.tag_id, tag.decision_margin, tag.hamming,
                                R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
                                t[0], t[1], t[2], pose_err
                            ])
                        else:
                            csv_writer_tag_right.writerow([
                                frame_out_idx, tag.tag_id, tag.decision_margin, tag.hamming,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            ])

            fps_text = f"{fps_ema:.1f} FPS"
            tag_text = f"Tags: L{len(tags_left)} R{len(tags_right)}" if apriltag_detector else ""
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            color = (0, 255, 0)
            org_y = 18

            if display_mode == "separate":
                if start_on_key and not recording:
                    cv2.putText(frame_l_vis, "Press S to start", (8, 42), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame_r_vis, "Press S to start", (8, 42), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame_l_vis, "SPACE: Photo", (8, 66), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame_r_vis, "SPACE: Photo", (8, 66), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                elif start_on_key and recording:
                    cv2.putText(frame_l_vis, "REC", (8, 42), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame_r_vis, "REC", (8, 42), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame_l_vis, "SPACE: Photo", (8, 66), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame_r_vis, "SPACE: Photo", (8, 66), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame_l_vis, "SPACE: Photo", (8, 42), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame_r_vis, "SPACE: Photo", (8, 42), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.putText(frame_l_vis, f"L {fps_text}", (8, org_y), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame_r_vis, f"R {fps_text}", (8, org_y), font, font_scale, color, thickness, cv2.LINE_AA)
                if tag_text:
                    cv2.putText(frame_l_vis, tag_text, (width - 120, org_y), font, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame_r_vis, tag_text, (width - 120, org_y), font, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow(left_window, frame_l_vis)
                cv2.imshow(right_window, frame_r_vis)

                # 保存视频：原始视频和识别后的视频
                if allow_record and save_video:
                    if vw_raw_l is None or vw_vis_l is None:
                        # 创建视频文件
                        raw_l = os.path.join(session_dir, "raw_left.mp4")
                        raw_r = os.path.join(session_dir, "raw_right.mp4")
                        vis_l = os.path.join(session_dir, "visualized_left.mp4")
                        vis_r = os.path.join(session_dir, "visualized_right.mp4")
                        
                        vw_raw_l = cv2.VideoWriter(raw_l, fourcc, actual_record_fps, (width, height))
                        vw_raw_r = cv2.VideoWriter(raw_r, fourcc, actual_record_fps, (width, height))
                        vw_vis_l = cv2.VideoWriter(vis_l, fourcc, actual_record_fps, (width, height))
                        vw_vis_r = cv2.VideoWriter(vis_r, fourcc, actual_record_fps, (width, height))
                        
                        if not vw_raw_l.isOpened() or not vw_vis_l.isOpened():
                            fourcc2 = cv2.VideoWriter_fourcc(*"XVID")
                            vw_raw_l = cv2.VideoWriter(raw_l, fourcc2, actual_record_fps, (width, height))
                            vw_raw_r = cv2.VideoWriter(raw_r, fourcc2, actual_record_fps, (width, height))
                            vw_vis_l = cv2.VideoWriter(vis_l, fourcc2, actual_record_fps, (width, height))
                            vw_vis_r = cv2.VideoWriter(vis_r, fourcc2, actual_record_fps, (width, height))

                    # 保存原始帧
                    if vw_raw_l is not None and vw_raw_l.isOpened():
                        vw_raw_l.write(frame_l_raw)
                    if vw_raw_r is not None and vw_raw_r.isOpened():
                        vw_raw_r.write(frame_r_raw)
                    
                    # 保存识别后的帧
                    if vw_vis_l is not None and vw_vis_l.isOpened():
                        vw_vis_l.write(frame_l_vis)
                    if vw_vis_r is not None and vw_vis_r.isOpened():
                        vw_vis_r.write(frame_r_vis)
            else:
                stitched = np.concatenate([frame_l_vis, frame_r_vis], axis=1)
                if start_on_key and not recording:
                    cv2.putText(stitched, "Press S to start", (8, 42), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(stitched, "SPACE: Photo | Q: Quit", (8, 66), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                elif start_on_key and recording:
                    cv2.putText(stitched, "REC", (8, 42), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(stitched, "SPACE: Photo | Q: Quit", (8, 66), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(stitched, "SPACE: Photo | Q: Quit", (8, 42), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(stitched, f"L {fps_text}", (8, org_y), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(stitched, f"R {fps_text}", (width + 8, org_y), font, font_scale, color, thickness, cv2.LINE_AA)
                if tag_text:
                    cv2.putText(stitched, tag_text, (width * 2 - 150, org_y), font, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow(stitched_window, stitched)

                # 保存视频：原始拼接视频和识别后的拼接视频
                if allow_record and save_video:
                    if vw_raw_stitch is None or vw_vis_stitch is None:
                        raw_stitch_path = os.path.join(session_dir, "raw_stereo.mp4")
                        vis_stitch_path = os.path.join(session_dir, "visualized_stereo.mp4")
                        
                        vw_raw_stitch = cv2.VideoWriter(raw_stitch_path, fourcc, actual_record_fps, (width * 2, height))
                        vw_vis_stitch = cv2.VideoWriter(vis_stitch_path, fourcc, actual_record_fps, (width * 2, height))
                        
                        if not vw_raw_stitch.isOpened() or not vw_vis_stitch.isOpened():
                            fourcc2 = cv2.VideoWriter_fourcc(*"XVID")
                            vw_raw_stitch = cv2.VideoWriter(raw_stitch_path, fourcc2, actual_record_fps, (width * 2, height))
                            vw_vis_stitch = cv2.VideoWriter(vis_stitch_path, fourcc2, actual_record_fps, (width * 2, height))

                    # 保存原始拼接帧
                    if vw_raw_stitch is not None and vw_raw_stitch.isOpened():
                        raw_stitched = np.concatenate([frame_l_raw, frame_r_raw], axis=1)
                        vw_raw_stitch.write(raw_stitched)
                    
                    # 保存识别后的拼接帧
                    if vw_vis_stitch is not None and vw_vis_stitch.isOpened():
                        vw_vis_stitch.write(stitched)

            key = cv2.waitKey(1) & 0xFF
            
            # 拍照功能 - 按空格键
            if key == 32:  # 空格键
                # 创建photos文件夹
                photos_dir = os.path.join(session_dir if recording else os.path.join(recordings_dir, "photos"))
                if not os.path.exists(photos_dir):
                    os.makedirs(photos_dir)
                
                # 生成时间戳文件名
                photo_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
                
                # 保存原始图像
                raw_left_path = os.path.join(photos_dir, f"photo_{photo_timestamp}_raw_left.jpg")
                raw_right_path = os.path.join(photos_dir, f"photo_{photo_timestamp}_raw_right.jpg")
                cv2.imwrite(raw_left_path, frame_l_raw)
                cv2.imwrite(raw_right_path, frame_r_raw)
                
                # 保存带关键点标注的图像
                vis_left_path = os.path.join(photos_dir, f"photo_{photo_timestamp}_vis_left.jpg")
                vis_right_path = os.path.join(photos_dir, f"photo_{photo_timestamp}_vis_right.jpg")
                cv2.imwrite(vis_left_path, frame_l_vis)
                cv2.imwrite(vis_right_path, frame_r_vis)
                
                # 保存拼接图像
                if display_mode == "stitch":
                    raw_stitch_path = os.path.join(photos_dir, f"photo_{photo_timestamp}_raw_stereo.jpg")
                    vis_stitch_path = os.path.join(photos_dir, f"photo_{photo_timestamp}_vis_stereo.jpg")
                    raw_stitched = np.concatenate([frame_l_raw, frame_r_raw], axis=1)
                    vis_stitched = np.concatenate([frame_l_vis, frame_r_vis], axis=1)
                    cv2.imwrite(raw_stitch_path, raw_stitched)
                    cv2.imwrite(vis_stitch_path, vis_stitched)
                
                # 保存2D关键点数据到单独的CSV
                if last_kpts_l is not None or last_kpts_r is not None:
                    csv_photo_path = os.path.join(photos_dir, f"photo_{photo_timestamp}_keypoints_2d.csv")
                    with open(csv_photo_path, "w", newline="", encoding="utf-8") as csv_photo:
                        csv_photo_writer = csv.writer(csv_photo)
                        csv_photo_writer.writerow(["camera", "kpt_idx", "kpt_name", "x", "y", "confidence"])
                        
                        if last_kpts_l is not None:
                            for idx in LOWER_KPT_IDXS:
                                if idx in last_kpts_l:
                                    x, y, conf = last_kpts_l[idx]
                                    csv_photo_writer.writerow(["left", idx, KPT_NAME.get(idx, str(idx)), float(x), float(y), float(conf)])
                        
                        if last_kpts_r is not None:
                            for idx in LOWER_KPT_IDXS:
                                if idx in last_kpts_r:
                                    x, y, conf = last_kpts_r[idx]
                                    csv_photo_writer.writerow(["right", idx, KPT_NAME.get(idx, str(idx)), float(x), float(y), float(conf)])
                
                print(f"📷 照片已保存到: {photos_dir}")
                print(f"   文件名: photo_{photo_timestamp}_*.jpg")
            
            # 录制控制
            if start_on_key and (not recording) and (key in (ord(start_key_norm), ord(start_key_norm.upper()))):
                recording = True
                record_frame_idx = 0
                
                # 创建会话文件夹
                if pending_session_dir:
                    session_dir = pending_session_dir
                    if not os.path.exists(session_dir):
                        os.makedirs(session_dir)
                
                # 创建CSV文件
                if pending_session_dir and csv_writer_left is None:
                    csv_path_left = os.path.join(session_dir, "keypoints_2d_left.csv")
                    csv_path_right = os.path.join(session_dir, "keypoints_2d_right.csv")
                    csv_f_left = open(csv_path_left, "w", newline="", encoding="utf-8")
                    csv_f_right = open(csv_path_right, "w", newline="", encoding="utf-8")
                    csv_writer_left = csv.writer(csv_f_left)
                    csv_writer_right = csv.writer(csv_f_right)
                    csv_writer_left.writerow(["frame", "kpt_idx", "kpt_name", "x", "y", "confidence"])
                    csv_writer_right.writerow(["frame", "kpt_idx", "kpt_name", "x", "y", "confidence"])
                    
                    # 创建AprilTag CSV文件
                    if apriltag_detector is not None:
                        csv_path_tag_left = os.path.join(session_dir, "apriltag_left.csv")
                        csv_path_tag_right = os.path.join(session_dir, "apriltag_right.csv")
                        csv_f_tag_left = open(csv_path_tag_left, "w", newline="", encoding="utf-8")
                        csv_f_tag_right = open(csv_path_tag_right, "w", newline="", encoding="utf-8")
                        csv_writer_tag_left = csv.writer(csv_f_tag_left)
                        csv_writer_tag_right = csv.writer(csv_f_tag_right)
                        header = ["frame", "tag_id", "decision_margin", "hamming",
                                 "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33",
                                 "tx", "ty", "tz", "pose_error"]
                        csv_writer_tag_left.writerow(header)
                        csv_writer_tag_right.writerow(header)
                
                print(f"🔴 录制开始 - 保存到: {session_dir}")
            
            # 退出
            if key in (27, ord("q")):
                break

    finally:
        left_cap.release()
        right_cap.release()
        cv2.destroyAllWindows()
        if csv_f_left is not None:
            csv_f_left.close()
        if csv_f_right is not None:
            csv_f_right.close()
        if csv_f_tag_left is not None:
            csv_f_tag_left.close()
        if csv_f_tag_right is not None:
            csv_f_tag_right.close()
        
        # 释放所有视频写入器
        if vw_raw_l is not None:
            vw_raw_l.release()
        if vw_raw_r is not None:
            vw_raw_r.release()
        if vw_raw_stitch is not None:
            vw_raw_stitch.release()
        if vw_vis_l is not None:
            vw_vis_l.release()
        if vw_vis_r is not None:
            vw_vis_r.release()
        if vw_vis_stitch is not None:
            vw_vis_stitch.release()
        
        if recording:
            print(f"\nRecording saved to: {session_dir}")
            print(f"  - Raw videos: raw_*.mp4")
            print(f"  - Visualized videos: visualized_*.mp4")
            if csv_f_left is not None:
                print(f"  - Left 2D keypoints: keypoints_2d_left.csv")
            if csv_f_right is not None:
                print(f"  - Right 2D keypoints: keypoints_2d_right.csv")
            if csv_f_tag_left is not None:
                print(f"  - Left AprilTag data: apriltag_left.csv")
            if csv_f_tag_right is not None:
                print(f"  - Right AprilTag data: apriltag_right.csv")



def _parse_source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="双目2D关键点录制程序")
    parser.add_argument("--left", type=_parse_source, default=DEFAULT_LEFT_CAMERA_ID, help="左相机索引或视频/url")
    parser.add_argument("--right", type=_parse_source, default=DEFAULT_RIGHT_CAMERA_ID, help="右相机索引或视频/url")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mode", choices=["separate", "stitch"], default="stitch", help="显示模式")
    parser.add_argument("--backend", choices=CAMERA_BACKEND_CHOICES, default="auto")
    parser.add_argument("--warmup-s", type=float, default=0.8)
    parser.add_argument("--drop-frames", type=int, default=20)
    parser.add_argument("--use-grab", action="store_true", default=True)
    parser.add_argument("--start-on-key", action="store_true", default=True)
    parser.add_argument("--start-key", type=str, default="s")
    parser.add_argument("--save-video", type=str, default="enabled")

    parser.add_argument("--enable-pose", action="store_true", default=True, help="启用mmdet+mmpose推理")
    parser.add_argument("--pose-side", choices=["left", "right", "both"], default="both", help="在哪个视图运行姿态估计")
    parser.add_argument("--infer-interval", type=int, default=1, help="每N帧运行一次姿态估计 (1=每帧推理/实时, 2=每2帧推理, 3=每3帧推理)")
    parser.add_argument("--device", type=str, default="auto", help="推理设备: auto、cuda:0 或 cpu")

    parser.add_argument("--show-names", action="store_true", default=True, help="显示关键点名称")
    parser.add_argument("--save-2d-csv", action="store_true", default=True, help="保存2D关键点CSV文件")
    
    # AprilTag参数
    parser.add_argument("--enable-apriltag", action="store_true", default=False, help="启用AprilTag检测")
    parser.add_argument("--apriltag-family", type=str, default="tag36h11", 
                       choices=["tag36h11", "tag25h9", "tag16h5", "tagStandard41h12", "tagStandard52h13"],
                       help="AprilTag家族")
    parser.add_argument("--apriltag-size", type=float, default=0.05, help="AprilTag物理尺寸(米)")
    parser.add_argument("--camera-config-yaml", type=str, default=None,
                       help="相机配置YAML文件路径（自动读取内参）")
    parser.add_argument("--camera-params-left", type=str, default=None, 
                       help="左相机内参 (fx,fy,cx,cy)，逗号分隔（优先级高于YAML）")
    parser.add_argument("--camera-params-right", type=str, default=None,
                       help="右相机内参 (fx,fy,cx,cy)，逗号分隔（优先级高于YAML）")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    det_config_default = os.path.join(script_dir, "demo", "mmdetection_cfg", "rtmdet_m_640-8xb32_coco-person.py")
    det_ckpt_default = os.path.join(script_dir, "checkpoints", "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")
    pose_config_default = os.path.join(
        script_dir, "configs", "body_2d_keypoint", "rtmpose", "body8",
        "rtmpose-m_8xb512-700e_body8-halpe26-256x192.py",
    )
    pose_ckpt_default = os.path.join(
        script_dir, "checkpoints",
        "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth",
    )

    parser.add_argument("--det-config", type=str, default=det_config_default)
    parser.add_argument("--det-checkpoint", type=str, default=det_ckpt_default)
    parser.add_argument("--pose-config", type=str, default=pose_config_default)
    parser.add_argument("--pose-checkpoint", type=str, default=pose_ckpt_default)

    parser.add_argument("--det-cat-id", type=int, default=0)
    parser.add_argument("--bbox-thr", type=float, default=0.5)
    parser.add_argument("--nms-thr", type=float, default=0.3)
    parser.add_argument("--kpt-thr", type=float, default=0.3)

    args = parser.parse_args()

    pose_runner: PoseRunner | None = None
    if args.enable_pose:
        pose_runner = PoseRunner(
            det_config=args.det_config,
            det_checkpoint=args.det_checkpoint,
            pose_config=args.pose_config,
            pose_checkpoint=args.pose_checkpoint,
            device=args.device,
            det_cat_id=args.det_cat_id,
            bbox_thr=args.bbox_thr,
            nms_thr=args.nms_thr,
            kpt_thr=args.kpt_thr,
            show_names=args.show_names,
        )
    
    # 解析相机内参
    camera_params_left = None
    camera_params_right = None
    
    # 方法1: 从YAML文件读取（如果提供）
    if args.camera_config_yaml and os.path.exists(args.camera_config_yaml):
        print(f"从YAML文件读取相机内参: {args.camera_config_yaml}")
        params_left_yaml, params_right_yaml = load_camera_params_from_yaml(args.camera_config_yaml)
        if params_left_yaml:
            camera_params_left = params_left_yaml
            print(f"  左相机内参: fx={camera_params_left[0]:.2f}, fy={camera_params_left[1]:.2f}, "
                  f"cx={camera_params_left[2]:.2f}, cy={camera_params_left[3]:.2f}")
        if params_right_yaml:
            camera_params_right = params_right_yaml
            print(f"  右相机内参: fx={camera_params_right[0]:.2f}, fy={camera_params_right[1]:.2f}, "
                  f"cx={camera_params_right[2]:.2f}, cy={camera_params_right[3]:.2f}")
    
    # 方法2: 从命令行参数读取（优先级更高，会覆盖YAML）
    if args.camera_params_left:
        try:
            camera_params_left = [float(x.strip()) for x in args.camera_params_left.split(',')]
            if len(camera_params_left) != 4:
                print("警告: 左相机内参格式错误，应为 fx,fy,cx,cy")
                camera_params_left = None
            else:
                print(f"使用命令行左相机内参: fx={camera_params_left[0]}, fy={camera_params_left[1]}, "
                      f"cx={camera_params_left[2]}, cy={camera_params_left[3]}")
        except:
            print("警告: 无法解析左相机内参")
            camera_params_left = None
    
    if args.camera_params_right:
        try:
            camera_params_right = [float(x.strip()) for x in args.camera_params_right.split(',')]
            if len(camera_params_right) != 4:
                print("警告: 右相机内参格式错误，应为 fx,fy,cx,cy")
                camera_params_right = None
            else:
                print(f"使用命令行右相机内参: fx={camera_params_right[0]}, fy={camera_params_right[1]}, "
                      f"cx={camera_params_right[2]}, cy={camera_params_right[3]}")
        except:
            print("警告: 无法解析右相机内参")
            camera_params_right = None
    
    # 如果启用AprilTag但没有相机内参，给出提示
    if args.enable_apriltag and (camera_params_left is None or camera_params_right is None):
        print("\n⚠️  警告: AprilTag检测已启用，但未提供相机内参")
        print("   将只能检测Tag位置，无法估计3D位姿")
        print("   提供内参的方法:")
        print("   1. 使用YAML文件: --camera-config-yaml configs/camera_config_xxx.yaml")
        print("   2. 使用命令行: --camera-params-left fx,fy,cx,cy --camera-params-right fx,fy,cx,cy\n")

    run_stereo_capture(
        left_source=args.left,
        right_source=args.right,
        width=args.width,
        height=args.height,
        fps=args.fps,
        display_mode=args.mode,
        backend=str(args.backend),
        pose_runner=pose_runner,
        pose_side=args.pose_side,
        infer_interval=args.infer_interval,
        kpt_thr=float(args.kpt_thr),
        save_2d_csv=bool(args.save_2d_csv),
        warmup_s=float(args.warmup_s),
        drop_frames=int(args.drop_frames),
        use_grab=bool(args.use_grab),
        start_on_key=bool(args.start_on_key),
        start_key=str(args.start_key),
        save_video=str(args.save_video),
        enable_apriltag=bool(args.enable_apriltag),
        apriltag_family=str(args.apriltag_family),
        apriltag_size=float(args.apriltag_size),
        camera_params_left=camera_params_left,
        camera_params_right=camera_params_right,
    )


if __name__ == "__main__":
    main()
