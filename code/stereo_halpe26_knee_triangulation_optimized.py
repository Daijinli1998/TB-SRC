'''这是一个双目相机实时姿态估计和3D重建系统 
相机初始化 → 预热/丢帧 → 主循环
  ├─ 读取左右帧
  ├─ 每N帧推理一次（infer_interval）
  ├─ 非推理帧：重绘上次关键点（防止闪烁）
  ├─ 双目三角化计算3D坐标
  ├─ 保存CSV/视频
  └─ 显示FPS和状态 '''

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np

from runtime_compat import (
    CAMERA_BACKEND_CHOICES,
    DEFAULT_LEFT_CAMERA_ID,
    DEFAULT_RIGHT_CAMERA_ID,
    open_stereo_captures,
    resolve_inference_device,
)

def _undistort_pixel(pt_xy: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    pts = np.array([[pt_xy.astype(np.float64)]], dtype=np.float64)
    out = cv2.undistortPoints(pts, K, D, P=K)
    return out.reshape(2).astype(np.float64)


def _rectify_pixel(pt_xy: np.ndarray, K: np.ndarray, D: np.ndarray, R: np.ndarray, P: np.ndarray) -> np.ndarray:
    pts = np.array([[pt_xy.astype(np.float64)]], dtype=np.float64)
    P_use = P
    if isinstance(P_use, np.ndarray) and P_use.shape == (3, 4):
        P_use = P_use[:, :3]
    out = cv2.undistortPoints(pts, K, D, R=R, P=P_use)
    return out.reshape(2).astype(np.float64)


def _triangulate_point(P1: np.ndarray, P2: np.ndarray, pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray | None:
    pts1 = pt1.reshape(2, 1).astype(np.float64)
    pts2 = pt2.reshape(2, 1).astype(np.float64)
    X_h = cv2.triangulatePoints(P1, P2, pts1, pts2)
    w = float(X_h[3, 0])
    if abs(w) < 1e-9:
        return None
    return (X_h[:3, 0] / w).astype(np.float64)


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
    width: int = 640,
    height: int = 480,
    fps: int = 60,
    display_mode: str = "stitch",
    backend: str = "auto",
    pose_runner: PoseRunner | None = None,
    pose_side: str = "left",
    infer_interval: int = 3,
    kpt_thr: float = 0.3,
    enable_3d: bool = False,
    camera_cfg=None,
    save_3d_csv: str = "",
    print_3d_every: int = 0,
    warmup_s: float = 0.0,
    drop_frames: int = 0,
    use_grab: bool = False,
    start_on_key: bool = False,
    start_key: str = "s",
    save_video: str = "",
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
    
    csv_f = None
    csv_writer = None
    pending_csv_path = ""
    pending_session_dir = ""
    
    if enable_3d and save_3d_csv:
        if start_on_key:
            pending_csv_path = save_3d_csv
            pending_session_dir = session_dir
        else:
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)
            csv_path = os.path.join(session_dir, "keypoints_3d.csv")
            csv_f = open(csv_path, "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_f)
            csv_writer.writerow(["frame", "kpt_idx", "kpt_name", "X", "Y", "Z", "conf_l", "conf_r"])

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

    # 优化：预分配3D计算所需的矩阵（如果启用3D）
    P1, P2, K1, D1, K2, D2, R1, R2 = None, None, None, None, None, None, None, None
    if enable_3d and camera_cfg is not None:
        P1 = camera_cfg.P1.astype(np.float64)
        P2 = camera_cfg.P2.astype(np.float64)
        K1 = camera_cfg.left_camera_matrix.astype(np.float64)
        D1 = camera_cfg.left_distortion.astype(np.float64)
        K2 = camera_cfg.right_camera_matrix.astype(np.float64)
        D2 = camera_cfg.right_distortion.astype(np.float64)
        R1 = camera_cfg.R1.astype(np.float64)
        R2 = camera_cfg.R2.astype(np.float64)

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
                if enable_3d and camera_cfg is not None and pose_side == "both":
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

            # 3D三角化计算
            if enable_3d and P1 is not None and last_kpts_l is not None and last_kpts_r is not None:
                pts3d = {}
                for idx in LOWER_KPT_IDXS:
                    vL = last_kpts_l.get(idx, None)
                    vR = last_kpts_r.get(idx, None)
                    if vL is None or vR is None:
                        pts3d[idx] = None
                        continue
                    xL, yL, cL = vL
                    xR, yR, cR = vR
                    if float(cL) < float(kpt_thr) or float(cR) < float(kpt_thr):
                        pts3d[idx] = None
                        continue

                    ptL = np.array([xL, yL], dtype=np.float64)
                    ptR = np.array([xR, yR], dtype=np.float64)
                    ptLr = _rectify_pixel(ptL, K1, D1, R1, P1)
                    ptRr = _rectify_pixel(ptR, K2, D2, R2, P2)
                    X = _triangulate_point(P1, P2, ptLr, ptRr)
                    pts3d[idx] = X

                    if allow_record and csv_writer is not None and X is not None:
                        csv_writer.writerow(
                            [frame_out_idx, idx, KPT_NAME.get(idx, str(idx)), 
                             float(X[0]), float(X[1]), float(X[2]), float(cL), float(cR)]
                        )

                if allow_record and print_3d_every and (frame_out_idx % int(print_3d_every) == 0):
                    msg_parts = []
                    for idx in LOWER_KPT_IDXS:
                        X = pts3d.get(idx, None)
                        if X is None:
                            continue
                        msg_parts.append(f"{KPT_NAME.get(idx, str(idx))}:({X[0]:.1f},{X[1]:.1f},{X[2]:.1f})")
                    if msg_parts:
                        print("3D "+" ".join(msg_parts))

            fps_text = f"{fps_ema:.1f} FPS"
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
                if pending_csv_path and csv_writer is None:
                    csv_path = os.path.join(session_dir, "keypoints_3d.csv")
                    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
                    csv_writer = csv.writer(csv_f)
                    csv_writer.writerow(["frame", "kpt_idx", "kpt_name", "X", "Y", "Z", "conf_l", "conf_r"])
                
                print(f"🔴 录制开始 - 保存到: {session_dir}")
            
            # 退出
            if key in (27, ord("q")):
                break

    finally:
        left_cap.release()
        right_cap.release()
        cv2.destroyAllWindows()
        if csv_f is not None:
            csv_f.close()
        
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
            if csv_f is not None:
                print(f"  - 3D keypoints: keypoints_3d.csv")



def _parse_source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="优化版双目相机捕获 + 姿态估计 + FPS显示")
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
    parser.add_argument("--save-video", type=str, default="enabled")  # 任何非空值都会启用保存

    parser.add_argument("--enable-pose", action="store_true", default=True, help="启用mmdet+mmpose推理")
    parser.add_argument("--pose-side", choices=["left", "right", "both"], default="both", help="在哪个视图运行姿态估计")
    parser.add_argument("--infer-interval", type=int, default=1, help="每N帧运行一次姿态估计")
    parser.add_argument("--device", type=str, default="auto", help="推理设备: auto、cuda:0 或 cpu")

    # 输出点名字标签开关
    parser.add_argument("--show-names", action="store_true", default=True, help="显示关键点名称")

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
    camera_config_default = os.path.join(script_dir, "configs", "camera_config_dechuang_640x480_LZ.yaml")

    parser.add_argument("--det-config", type=str, default=det_config_default)
    parser.add_argument("--det-checkpoint", type=str, default=det_ckpt_default)
    parser.add_argument("--pose-config", type=str, default=pose_config_default)
    parser.add_argument("--pose-checkpoint", type=str, default=pose_ckpt_default)

    parser.add_argument("--det-cat-id", type=int, default=0)
    parser.add_argument("--bbox-thr", type=float, default=0.5)
    parser.add_argument("--nms-thr", type=float, default=0.3)
    parser.add_argument("--kpt-thr", type=float, default=0.3)

    parser.add_argument("--enable-3d", action="store_true", default=True)
    parser.add_argument("--camera-config", type=str, default=camera_config_default)
    parser.add_argument("--save-3d-csv", type=str, default="out_3d.csv")
    parser.add_argument("--print-3d-every", type=int, default=0)
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

    camera_cfg = None
    if args.enable_3d:
        try:
            from calibration.CameraConfig import CameraConfig
        except Exception as e:
            raise RuntimeError("Failed to import calibration.CameraConfig (and its dependencies)") from e

        camera_config_path = args.camera_config
        if not camera_config_path:
            camera_config_path = os.path.join(script_dir, "configs", "camera_config_dechuang_640x480_ttt.yaml")
        if not os.path.exists(camera_config_path):
            raise FileNotFoundError(f"camera config not found: {camera_config_path}")

        camera_cfg = CameraConfig(camera_config_path)
        if getattr(camera_cfg, "P1", None) is None or getattr(camera_cfg, "P2", None) is None:
            raise RuntimeError("CameraConfig missing P1/P2. Check calibration yaml.")

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
        enable_3d=bool(args.enable_3d),
        camera_cfg=camera_cfg,
        save_3d_csv=str(args.save_3d_csv),
        print_3d_every=int(args.print_3d_every),
        warmup_s=float(args.warmup_s),
        drop_frames=int(args.drop_frames),
        use_grab=bool(args.use_grab),
        start_on_key=bool(args.start_on_key),
        start_key=str(args.start_key),
        save_video=str(args.save_video),
    )


if __name__ == "__main__":
    main()
