import argparse
import csv
import os
import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

# Halpe26 下半身关键点索引 (9个点)
# 11: left_hip, 12: right_hip, 13: left_knee, 14: right_knee
# 15: left_ankle, 16: right_ankle, 19: hip(中心), 20: left_big_toe, 21: right_big_toe
LOWER_KPT_IDXS = [11, 12, 13, 14, 15, 16, 19, 20, 21]
KPT_NAME = {
    11: "lhip", 12: "rhip", 13: "lkne", 14: "rkne",
    15: "lank", 16: "rank", 19: "hip", 20: "ltoe", 21: "rtoe"
}

# 骨架连接
LOWER_EDGES = [
    (15, 13), (13, 11),  # 左腿: 踝->膝->髋
    (16, 14), (14, 12),  # 右腿: 踝->膝->髋
    (11, 19), (12, 19),  # 左右髋->中心髋
    (15, 20), (16, 21)   # 踝->脚趾
]

def _draw_lower_only(img, kpts_xy, kpts_conf, kpt_thr):
    for a, b in LOWER_EDGES:
        if kpts_conf[a] > kpt_thr and kpts_conf[b] > kpt_thr:
            p1 = (int(kpts_xy[a][0]), int(kpts_xy[a][1]))
            p2 = (int(kpts_xy[b][0]), int(kpts_xy[b][1]))
            cv2.line(img, p1, p2, (0, 255, 0), 2)
    for idx in LOWER_KPT_IDXS:
        if kpts_conf[idx] > kpt_thr:
            p = (int(kpts_xy[idx][0]), int(kpts_xy[idx][1]))
            cv2.circle(img, p, 4, (0, 0, 255), -1)
            cv2.putText(img, KPT_NAME[idx], (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

class PoseRunner:
    def __init__(self, det_cfg, det_ckpt, pose_cfg, pose_ckpt, device='cuda:0'):
        self.detector = init_detector(det_cfg, det_ckpt, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        self.pose_estimator = init_pose_estimator(pose_cfg, pose_ckpt, device=device)

    def process_frame(self, frame, bbox_thr=0.5, kpt_thr=0.3):
        det_result = inference_detector(self.detector, frame)
        pred_instances = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate((pred_instances.bboxes, pred_instances.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instances.labels == 0, pred_instances.scores > bbox_thr)]
        if bboxes.shape[0] == 0: return frame, None
        keep = nms(bboxes, 0.3)
        bboxes = bboxes[keep, :4]
        
        pose_results = inference_topdown(self.pose_estimator, frame, bboxes)
        data_samples = merge_data_samples(pose_results)
        pred = data_samples.get("pred_instances", None)
        if pred is None: return frame, None
        
        kpts = pred.keypoints[0]
        conf = pred.keypoint_scores[0]
        vis_frame = frame.copy()
        _draw_lower_only(vis_frame, kpts, conf, kpt_thr)
        return vis_frame, {idx: (float(kpts[idx][0]), float(kpts[idx][1]), float(conf[idx])) for idx in LOWER_KPT_IDXS}

def process_recording(session_path, runner, output_root):
    # 查找视频文件
    session_name = os.path.basename(session_path)
    video_file = None
    
    # 优先查找固定名称的视频文件
    for f in ['stereo.mp4', 'raw_stereo.mp4', 'raw_video.mp4']:
        candidate = os.path.join(session_path, f)
        if os.path.exists(candidate):
            video_file = candidate
            break
    
    # 如果没找到，查找与文件夹同名的视频文件
    if not video_file:
        candidate = os.path.join(session_path, f'{session_name}.mp4')
        if os.path.exists(candidate):
            video_file = candidate
    
    # 如果还没找到，查找文件夹内任意 .mp4 文件
    if not video_file:
        for f in os.listdir(session_path):
            if f.endswith('.mp4'):
                video_file = os.path.join(session_path, f)
                break
    
    if not video_file:
        print(f"[{session_name}] No video file found, skipping")
        return

    out_dir = os.path.join(output_root, session_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    half_w = w // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw_stereo = cv2.VideoWriter(os.path.join(out_dir, 'visualized_stereo.mp4'), fourcc, fps, (w, h))
    
    csv_l = open(os.path.join(out_dir, 'keypoints_2d_left.csv'), 'w', newline='')
    csv_r = open(os.path.join(out_dir, 'keypoints_2d_right.csv'), 'w', newline='')
    writer_l = csv.writer(csv_l); writer_r = csv.writer(csv_r)
    header = ["frame", "kpt_idx", "kpt_name", "x", "y", "conf"]
    writer_l.writerow(header); writer_r.writerow(header)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        fl, fr = frame[:, :half_w], frame[:, half_w:]
        vl, kl = runner.process_frame(fl)
        vr, kr = runner.process_frame(fr)
        
        # 拼接左右视频
        stereo_vis = np.hstack([vl, vr])
        vw_stereo.write(stereo_vis)
        
        if kl: 
            for i, (x, y, c) in kl.items(): writer_l.writerow([frame_idx, i, KPT_NAME[i], x, y, c])
        if kr: 
            for i, (x, y, c) in kr.items(): writer_r.writerow([frame_idx, i, KPT_NAME[i], x, y, c])
        frame_idx += 1
        if frame_idx % 50 == 0: print(f"[{session_name}] Processed {frame_idx} frames")

    cap.release(); vw_stereo.release(); csv_l.close(); csv_r.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="recordings")
    parser.add_argument("--output", default="recordings_analysis")
    parser.add_argument("--session", help="Specific session folder name")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    det_cfg = os.path.join(script_dir, "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py")
    det_ckpt = os.path.join(script_dir, "checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")
    pose_cfg = os.path.join(script_dir, "configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py")
    pose_ckpt = os.path.join(script_dir, "checkpoints/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth")

    runner = PoseRunner(det_cfg, det_ckpt, pose_cfg, pose_ckpt)
    
    if args.session:
        process_recording(os.path.join(args.input, args.session), runner, args.output)
    else:
        for session in os.listdir(args.input):
            session_path = os.path.join(args.input, session)
            if os.path.isdir(session_path):
                process_recording(session_path, runner, args.output)
