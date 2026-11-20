import argparse
import csv
import os
import time
from datetime import datetime
from typing import Optional
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
LOWER_KPT_IDXS = [11, 12, 13, 14, 15, 16,  20, 21 ] #19，24, 25 脚跟参数，需要再加入
KPT_NAME = {
    11: "lhip",
    12: "rhip",
    13: "lkne",
    14: "rkne",
    15: "lank",
    16: "rank",
    #19: "hip",
    20: "ltoe",
    21: "rtoe",
    #24: "lheel",
    #25: "rheel",
}

# 简单下肢骨架连线（按需可加）
LOWER_EDGES = [
    (11, 13), (13, 15),             # left hip-knee-ankle 左腿
    (12, 14), (14, 16),             # right hip-knee-ankle 右腿
    (15, 20),  #左脚踝连脚趾                     #(15, 24), (24, 20), (20, 15),   # left ankle -> heel  左脚踝
    (16, 21)   #右脚踝连脚趾                    #(16, 25), (25, 21), (21, 16),   # right ankle -> toe/heel 右脚踝
]


def _draw_lower_only(
    img: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    kpt_thr: float,
    show_names: bool = True,
) -> np.ndarray:
    """只画下半身指定关键点与连线(OpenCV画,不用mmpose visualizer)。"""
    vis = img

    # points
    for idx in LOWER_KPT_IDXS:
        c = float(kpts_conf[idx])
        if c < kpt_thr:
            continue
        x, y = kpts_xy[idx]
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if not (0 <= xi < vis.shape[1] and 0 <= yi < vis.shape[0]):
            continue

        r = 5 if idx in (11, 12, 13, 14, 15, 16) else 4
        cv2.circle(vis, (xi, yi), r, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        if show_names:
            name = KPT_NAME.get(idx, str(idx))
            cv2.putText(
                vis,
                name,
                (min(xi + 6, vis.shape[1] - 1), max(yi - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # edges
    for a, b in LOWER_EDGES:
        ca = float(kpts_conf[a])
        cb = float(kpts_conf[b])
        if ca < kpt_thr or cb < kpt_thr:
            continue
        x1, y1 = kpts_xy[a]
        x2, y2 = kpts_xy[b]
        p1 = (int(round(float(x1))), int(round(float(y1))))
        p2 = (int(round(float(x2))), int(round(float(y2))))
        if not (0 <= p1[0] < vis.shape[1] and 0 <= p1[1] < vis.shape[0]):
            continue
        if not (0 <= p2[0] < vis.shape[1] and 0 <= p2[1] < vis.shape[0]):
            continue
        cv2.line(vis, p1, p2, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    return vis


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

    def visualize_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        """返回只画下半身关键点的 BGR 图（不画全身）。"""
        with self._torch.no_grad():
            det_result = self._inference_detector(self.detector, frame_bgr)
            pred_instance = det_result.pred_instances.cpu().numpy()

            # xyxy + score => [N,5]
            bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[
                np.logical_and(pred_instance.labels == self._det_cat_id, pred_instance.scores > self._bbox_thr)
            ]
            if bboxes.shape[0] == 0:
                return frame_bgr

            keep = self._nms(bboxes, self._nms_thr)
            bboxes = bboxes[keep, :]
            if bboxes.shape[0] == 0:
                return frame_bgr

            # 只取主目标 bbox（面积*score 最大）
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            scores = bboxes[:, 4]
            best = int(np.argmax(areas * scores))
            bboxes = bboxes[best:best + 1, :4]  # [1,4]

            pose_results = self._inference_topdown(self.pose_estimator, frame_bgr, bboxes)
            data_samples = self._merge_data_samples(pose_results)

        pred = data_samples.get("pred_instances", None)
        if pred is None:
            return frame_bgr

        kpts = getattr(pred, "keypoints", None)
        kpt_scores = getattr(pred, "keypoint_scores", None)

        if kpts is None:
            return frame_bgr

        # kpts: [N,K,2]  kpt_scores: [N,K]
        if isinstance(kpts, np.ndarray):
            kpts_np = kpts
        else:
            kpts_np = kpts.detach().cpu().numpy()

        if kpts_np.ndim != 3 or kpts_np.shape[0] == 0:
            return frame_bgr

        k0 = kpts_np[0]  # [K,2]

        if kpt_scores is None:
            c0 = np.ones((k0.shape[0],), dtype=np.float32)
        else:
            if isinstance(kpt_scores, np.ndarray):
                c_np = kpt_scores
            else:
                c_np = kpt_scores.detach().cpu().numpy()
            c0 = c_np[0] if (c_np.ndim == 2 and c_np.shape[0] > 0) else np.ones((k0.shape[0],), dtype=np.float32)

        vis = frame_bgr.copy()
        vis = _draw_lower_only(vis, k0, c0, self._kpt_thr, show_names=self._show_names)
        return vis

    def get_lower_kpts(self, frame_bgr: np.ndarray):
        """
        可选：如果你后续要“只输出下半身坐标”，用这个函数更方便。
        返回 dict: {idx: (x,y,conf)}，没有人则返回 None
        """
        with self._torch.no_grad():
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

            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            scores = bboxes[:, 4]
            best = int(np.argmax(areas * scores))
            bboxes = bboxes[best:best + 1, :4]

            pose_results = self._inference_topdown(self.pose_estimator, frame_bgr, bboxes)
            data_samples = self._merge_data_samples(pose_results)

        pred = data_samples.get("pred_instances", None)
        if pred is None:
            return None

        kpts = getattr(pred, "keypoints", None)
        kpt_scores = getattr(pred, "keypoint_scores", None)
        if kpts is None:
            return None

        if not isinstance(kpts, np.ndarray):
            kpts = kpts.detach().cpu().numpy()
        k0 = kpts[0]  # [K,2]

        if kpt_scores is None:
            c0 = np.ones((k0.shape[0],), dtype=np.float32)
        else:
            if not isinstance(kpt_scores, np.ndarray):
                kpt_scores = kpt_scores.detach().cpu().numpy()
            c0 = kpt_scores[0]

        out = {}
        for idx in LOWER_KPT_IDXS:
            x, y = float(k0[idx][0]), float(k0[idx][1])
            conf = float(c0[idx])
            out[idx] = (x, y, conf)
        return out

    def visualize_and_get_lower_kpts(self, frame_bgr: np.ndarray):
        with self._torch.no_grad():
            det_result = self._inference_detector(self.detector, frame_bgr)
            pred_instance = det_result.pred_instances.cpu().numpy()

            bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[
                np.logical_and(pred_instance.labels == self._det_cat_id, pred_instance.scores > self._bbox_thr)
            ]
            if bboxes.shape[0] == 0:
                return frame_bgr, None

            keep = self._nms(bboxes, self._nms_thr)
            bboxes = bboxes[keep, :]
            if bboxes.shape[0] == 0:
                return frame_bgr, None

            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            scores = bboxes[:, 4]
            best = int(np.argmax(areas * scores))
            bboxes = bboxes[best:best + 1, :4]

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

        vis = frame_bgr.copy()
        vis = _draw_lower_only(vis, k0, c0, self._kpt_thr, show_names=self._show_names)

        out = {}
        for idx in LOWER_KPT_IDXS:
            x, y = float(k0[idx][0]), float(k0[idx][1])
            conf = float(c0[idx])
            out[idx] = (x, y, conf)
        return vis, out


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
    infer_interval: int = 1,
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
    last_vis_l: np.ndarray | None = None
    last_vis_r: np.ndarray | None = None
    last_kpts_l = None
    last_kpts_r = None

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

    grab_fallback_to_read = False

    use_grab = True

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
            if do_infer:
                if enable_3d and camera_cfg is not None and pose_side == "both":
                    last_vis_l, last_kpts_l = pose_runner.visualize_and_get_lower_kpts(frame_l)
                    last_vis_r, last_kpts_r = pose_runner.visualize_and_get_lower_kpts(frame_r)
                else:
                    last_kpts_l = None
                    last_kpts_r = None

                    if pose_side in {"left", "both"}:
                        last_vis_l = pose_runner.visualize_bgr(frame_l)
                    else:
                        last_vis_l = frame_l

                    if pose_side in {"right", "both"}:
                        last_vis_r = pose_runner.visualize_bgr(frame_r)
                    else:
                        last_vis_r = frame_r
            else:
                if last_vis_l is None:
                    last_vis_l = frame_l
                if last_vis_r is None:
                    last_vis_r = frame_r

            frame_l_show = last_vis_l
            frame_r_show = last_vis_r

            allow_record = bool(recording)
            if allow_record:
                record_frame_idx += 1
            frame_out_idx = int(record_frame_idx) if start_on_key else int(frame_idx)

            if enable_3d and camera_cfg is not None and last_kpts_l is not None and last_kpts_r is not None:
                P1 = camera_cfg.P1.astype(np.float64)
                P2 = camera_cfg.P2.astype(np.float64)
                K1 = camera_cfg.left_camera_matrix.astype(np.float64)
                D1 = camera_cfg.left_distortion.astype(np.float64)
                K2 = camera_cfg.right_camera_matrix.astype(np.float64)
                D2 = camera_cfg.right_distortion.astype(np.float64)
                R1 = camera_cfg.R1.astype(np.float64)
                R2 = camera_cfg.R2.astype(np.float64)

                pts3d = {}
                for idx in LOWER_KPT_IDXS:
                    vL = last_kpts_l.get(idx, None)
                    vR = last_kpts_r.get(idx, None)
                    if vL is None or vR is None:
                        pts3d[idx] = None
                        continue
                    xL, yL, cL = vL
                    xR, yR, cR = vR
                    if float(cL) < float(kpt_thr):
                        pts3d[idx] = None
                        continue
                    if float(cR) < float(kpt_thr):
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
                            [frame_out_idx, idx, KPT_NAME.get(idx, str(idx)), float(X[0]), float(X[1]), float(X[2]), float(cL), float(cR)]
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
                    cv2.putText(frame_l_show, "Press S to start", (8, 42), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame_r_show, "Press S to start", (8, 42), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                elif start_on_key and recording:
                    cv2.putText(frame_l_show, "REC", (8, 42), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame_r_show, "REC", (8, 42), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.putText(frame_l_show, f"L {fps_text}", (8, org_y), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame_r_show, f"R {fps_text}", (8, org_y), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.imshow(left_window, frame_l_show)
                cv2.imshow(right_window, frame_r_show)

                # 保存视频：原始视频和识别后的视频
                if allow_record and save_video:
                    if vw_raw_l is None or vw_vis_l is None:
                        # 创建视频文件
                        raw_l = os.path.join(session_dir, "raw_left.mp4")
                        raw_r = os.path.join(session_dir, "raw_right.mp4")
                        vis_l = os.path.join(session_dir, "visualized_left.mp4")
                        vis_r = os.path.join(session_dir, "visualized_right.mp4")
                        
                        vw_raw_l = cv2.VideoWriter(raw_l, fourcc, float(fps) if fps > 0 else 30.0, (width, height))
                        vw_raw_r = cv2.VideoWriter(raw_r, fourcc, float(fps) if fps > 0 else 30.0, (width, height))
                        vw_vis_l = cv2.VideoWriter(vis_l, fourcc, float(fps) if fps > 0 else 30.0, (width, height))
                        vw_vis_r = cv2.VideoWriter(vis_r, fourcc, float(fps) if fps > 0 else 30.0, (width, height))
                        
                        if not vw_raw_l.isOpened() or not vw_vis_l.isOpened():
                            fourcc2 = cv2.VideoWriter_fourcc(*"XVID")
                            vw_raw_l = cv2.VideoWriter(raw_l, fourcc2, float(fps) if fps > 0 else 30.0, (width, height))
                            vw_raw_r = cv2.VideoWriter(raw_r, fourcc2, float(fps) if fps > 0 else 30.0, (width, height))
                            vw_vis_l = cv2.VideoWriter(vis_l, fourcc2, float(fps) if fps > 0 else 30.0, (width, height))
                            vw_vis_r = cv2.VideoWriter(vis_r, fourcc2, float(fps) if fps > 0 else 30.0, (width, height))

                    # 保存原始帧
                    if vw_raw_l is not None and vw_raw_l.isOpened():
                        vw_raw_l.write(frame_l)
                    if vw_raw_r is not None and vw_raw_r.isOpened():
                        vw_raw_r.write(frame_r)
                    
                    # 保存识别后的帧
                    if vw_vis_l is not None and vw_vis_l.isOpened():
                        vw_vis_l.write(frame_l_show)
                    if vw_vis_r is not None and vw_vis_r.isOpened():
                        vw_vis_r.write(frame_r_show)
            else:
                stitched = np.concatenate([frame_l_show, frame_r_show], axis=1)
                if start_on_key and not recording:
                    cv2.putText(stitched, "Press S to start", (8, 42), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                elif start_on_key and recording:
                    cv2.putText(stitched, "REC", (8, 42), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(stitched, f"L {fps_text}", (8, org_y), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(stitched, f"R {fps_text}", (width + 8, org_y), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.imshow(stitched_window, stitched)

                # 保存视频：原始拼接视频和识别后的拼接视频
                if allow_record and save_video:
                    if vw_raw_stitch is None or vw_vis_stitch is None:
                        raw_stitch_path = os.path.join(session_dir, "raw_stereo.mp4")
                        vis_stitch_path = os.path.join(session_dir, "visualized_stereo.mp4")
                        
                        vw_raw_stitch = cv2.VideoWriter(raw_stitch_path, fourcc, float(fps) if fps > 0 else 30.0, (width * 2, height))
                        vw_vis_stitch = cv2.VideoWriter(vis_stitch_path, fourcc, float(fps) if fps > 0 else 30.0, (width * 2, height))
                        
                        if not vw_raw_stitch.isOpened() or not vw_vis_stitch.isOpened():
                            fourcc2 = cv2.VideoWriter_fourcc(*"XVID")
                            vw_raw_stitch = cv2.VideoWriter(raw_stitch_path, fourcc2, float(fps) if fps > 0 else 30.0, (width * 2, height))
                            vw_vis_stitch = cv2.VideoWriter(vis_stitch_path, fourcc2, float(fps) if fps > 0 else 30.0, (width * 2, height))

                    # 保存原始拼接帧
                    if vw_raw_stitch is not None and vw_raw_stitch.isOpened():
                        raw_stitched = np.concatenate([frame_l, frame_r], axis=1)
                        vw_raw_stitch.write(raw_stitched)
                    
                    # 保存识别后的拼接帧
                    if vw_vis_stitch is not None and vw_vis_stitch.isOpened():
                        vw_vis_stitch.write(stitched)

            key = cv2.waitKey(1) & 0xFF
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
                
                print(f"Recording started - Saving to: {session_dir}")
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
    parser = argparse.ArgumentParser(description="Stereo camera capture + pose (optional) + FPS overlay")
    parser.add_argument("--left", type=_parse_source, default=DEFAULT_LEFT_CAMERA_ID, help="Left camera index or video/url")
    parser.add_argument("--right", type=_parse_source, default=DEFAULT_RIGHT_CAMERA_ID, help="Right camera index or video/url")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mode", choices=["separate", "stitch"], default="stitch", help="Display mode")
    parser.add_argument("--backend", choices=CAMERA_BACKEND_CHOICES, default="auto")
    parser.add_argument("--warmup-s", type=float, default=0.8)
    parser.add_argument("--drop-frames", type=int, default=20)
    parser.add_argument("--use-grab", action="store_true", default=True)
    parser.add_argument("--start-on-key", action="store_true", default=True)
    parser.add_argument("--start-key", type=str, default="s")
    parser.add_argument("--save-video", type=str, default="enabled")  # 任何非空值都会启用保存

    parser.add_argument("--enable-pose", action="store_true", default=True, help="Enable mmdet+mmpose inference")
    parser.add_argument("--pose-side", choices=["left", "right", "both"], default="both", help="Which view to run pose")
    parser.add_argument("--infer-interval", type=int, default=2, help="Run pose every N frames")
    parser.add_argument("--device", type=str, default="auto", help="Inference device: auto, cuda:0, or cpu")

    # 输出点名字标签开关
    parser.add_argument("--show-names", action="store_true", default=True, help="Show keypoint short names")

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
