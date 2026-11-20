import argparse
import csv
import os
import time
from datetime import datetime
import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

# Halpe26 下半身点：髋/膝/踝 + 大脚趾/脚跟
# left:  11 13 15 19 21
# right: 12 14 16 22 24
# 19 是 hip 中心点
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

LOWER_EDGES = [
    (11, 13), (13, 15),  # left hip-knee-ankle
    (12, 14), (14, 16),  # right hip-knee-ankle
    (15, 20),            # 左脚踝连脚趾
    (16, 21)             # 右脚踝连脚趾
]

def _draw_lower_only(img, kpts_xy, kpts_conf, kpt_thr, show_names=True):
    for a, b in LOWER_EDGES:
        ca, cb = float(kpts_conf[a]), float(kpts_conf[b])
        if ca < kpt_thr or cb < kpt_thr: continue
        p1 = (int(round(float(kpts_xy[a][0]))), int(round(float(kpts_xy[a][1]))))
        p2 = (int(round(float(kpts_xy[b][0]))), int(round(float(kpts_xy[b][1]))))
        if 0 <= p1[0] < img.shape[1] and 0 <= p1[1] < img.shape[0] and \
           0 <= p2[0] < img.shape[1] and 0 <= p2[1] < img.shape[0]:
            cv2.line(img, p1, p2, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    for idx in LOWER_KPT_IDXS:
        c = float(kpts_conf[idx])
        if c < kpt_thr: continue
        x, y = kpts_xy[idx]
        xi, yi = int(round(float(x))), int(round(float(y)))
        if not (0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]): continue
        r = 3 if idx in (11, 12, 13, 14, 15, 16, 19) else 2
        cv2.circle(img, (xi, yi), r, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        if show_names:
            name = KPT_NAME.get(idx, str(idx))
            cv2.putText(img, name, (min(xi + 5, img.shape[1] - 1), max(yi - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return img

class PoseRunner:
    def __init__(self, det_config, det_checkpoint, pose_config, pose_checkpoint, device='cuda:0', 
                 bbox_thr=0.5, nms_thr=0.3, kpt_thr=0.3):
        self.detector = init_detector(det_config, det_checkpoint, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        self.pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device=device)
        self.bbox_thr = bbox_thr
        self.nms_thr = nms_thr
        self.kpt_thr = kpt_thr

    def process_frame(self, frame_bgr):
        with torch.no_grad():
            det_result = inference_detector(self.detector, frame_bgr)
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > self.bbox_thr)]
            if bboxes.shape[0] == 0: return frame_bgr, None
            keep = nms(bboxes, self.nms_thr)
            bboxes = bboxes[keep, :4]
            if bboxes.shape[0] == 0: return frame_bgr, None
            
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            best = int(np.argmax(areas))
            bboxes = bboxes[best:best + 1, :4]

            pose_results = inference_topdown(self.pose_estimator, frame_bgr, bboxes)
            data_samples = merge_data_samples(pose_results)
            pred = data_samples.get("pred_instances", None)
            if pred is None: return frame_bgr, None

            kpts = pred.keypoints[0]
            kpt_scores = pred.keypoint_scores[0]
            vis_frame = frame_bgr.copy()
            _draw_lower_only(vis_frame, kpts, kpt_scores, self.kpt_thr)
            
            out = {idx: (float(kpts[idx][0]), float(kpts[idx][1]), float(kpt_scores[idx])) for idx in LOWER_KPT_IDXS}
            return vis_frame, out

def process_video(video_path, output_dir, pose_runner):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 假设左右拼接视频，宽度减半
    half_width = width // 2
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw_l = cv2.VideoWriter(os.path.join(output_dir, 'left_vis.mp4'), fourcc, fps, (half_width, height))
    vw_r = cv2.VideoWriter(os.path.join(output_dir, 'right_vis.mp4'), fourcc, fps, (half_width, height))
    
    csv_l_f = open(os.path.join(output_dir, 'keypoints_2d_left.csv'), 'w', newline='', encoding='utf-8')
    csv_r_f = open(os.path.join(output_dir, 'keypoints_2d_right.csv'), 'w', newline='', encoding='utf-8')
    writer_l = csv.writer(csv_l_f)
    writer_r = csv.writer(csv_r_f)
    header = ["frame", "kpt_idx", "kpt_name", "x", "y", "confidence"]
    writer_l.writerow(header)
    writer_r.writerow(header)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_l = frame[:, :half_width]
        frame_r = frame[:, half_width:]
        
        vis_l, kpts_l = pose_runner.process_frame(frame_l)
        vis_r, kpts_r = pose_runner.process_frame(frame_r)
        
        vw_l.write(vis_l)
        vw_r.write(vis_r)
        
        if kpts_l:
            for idx, (x, y, conf) in kpts_l.items():
                writer_l.writerow([frame_idx, idx, KPT_NAME[idx], x, y, conf])
        if kpts_r:
            for idx, (x, y, conf) in kpts_r.items():
                writer_r.writerow([frame_idx, idx, KPT_NAME[idx], x, y, conf])
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    vw_l.release()
    vw_r.release()
    csv_l_f.close()
    csv_r_f.close()
    print(f"Finished processing. Results saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="output_video_analysis")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    det_config = os.path.join(script_dir, "demo", "mmdetection_cfg", "rtmdet_m_640-8xb32_coco-person.py")
    det_ckpt = os.path.join(script_dir, "checkpoints", "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")
    pose_config = os.path.join(script_dir, "configs", "body_2d_keypoint", "rtmpose", "body8", "rtmpose-m_8xb512-700e_body8-halpe26-256x192.py")
    pose_ckpt = os.path.join(script_dir, "checkpoints", "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth")

    runner = PoseRunner(det_config, det_ckpt, pose_config, pose_ckpt, device=args.device)
    process_video(args.video, args.output, runner)
