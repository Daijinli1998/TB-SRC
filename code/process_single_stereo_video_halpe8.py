"""
使用 Halpe8 (8个下半身关键点) 权重处理单个拼接视频
输出: 左右目2D关键点CSV + 拼接可视化视频
"""
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

# Halpe8 下半身关键点索引 (基于 checkpoints/halpe8.py)
LOWER_KPT_IDXS = [0, 1, 2, 3, 4, 5, 6, 7]
LEGACY_KPT_NAME = {
    0: "rhip", 1: "lhip", 2: "rkne", 3: "lkne",  # 左右对调
    4: "rank", 5: "lank", 6: "rtoe", 7: "ltoe"   # 左右对调
}
STANDARD_KPT_NAME = {
    0: "lhip", 1: "rhip", 2: "lkne", 3: "rkne",
    4: "lank", 5: "rank", 6: "ltoe", 7: "rtoe",
}

# 骨架连接 (左右对调后)
LOWER_EDGES = [
    (5, 3), (3, 1),  # 左腿: 踝->膝->髋 (实际是5,3,1)
    (4, 2), (2, 0),  # 右腿: 踝->膝->髋 (实际是4,2,0)
    (0, 1),          # 左右髋连接
    (5, 7), (4, 6)   # 踝->脚趾 (左右对调)
]

def _draw_lower_only(img, kpts_xy, kpts_conf, kpt_thr, kpt_name):
    """绘制下半身关键点和骨架"""
    for a, b in LOWER_EDGES:
        if kpts_conf[a] > kpt_thr and kpts_conf[b] > kpt_thr:
            p1 = (int(kpts_xy[a][0]), int(kpts_xy[a][1]))
            p2 = (int(kpts_xy[b][0]), int(kpts_xy[b][1]))
            cv2.line(img, p1, p2, (0, 255, 0), 2)
    for idx in LOWER_KPT_IDXS:
        if kpts_conf[idx] > kpt_thr:
            p = (int(kpts_xy[idx][0]), int(kpts_xy[idx][1]))
            cv2.circle(img, p, 4, (0, 0, 255), -1)
            cv2.putText(img, kpt_name[idx], (p[0]+5, p[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

class PoseRunner:
    def __init__(
        self,
        det_cfg,
        det_ckpt,
        pose_cfg,
        pose_ckpt,
        device='cuda:0',
        legacy_swapped_order=True,
    ):
        self.detector = init_detector(det_cfg, det_ckpt, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        self.pose_estimator = init_pose_estimator(pose_cfg, pose_ckpt, device=device)
        self.kpt_name = LEGACY_KPT_NAME if legacy_swapped_order else STANDARD_KPT_NAME

    def process_frame(self, frame, bbox_thr=0.5, kpt_thr=0.3):
        det_result = inference_detector(self.detector, frame)
        pred_instances = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate((pred_instances.bboxes, pred_instances.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instances.labels == 0, pred_instances.scores > bbox_thr)]
        if bboxes.shape[0] == 0: 
            return frame, None
        keep = nms(bboxes, 0.3)
        bboxes = bboxes[keep, :4]
        
        pose_results = inference_topdown(self.pose_estimator, frame, bboxes)
        data_samples = merge_data_samples(pose_results)
        pred = data_samples.get("pred_instances", None)
        if pred is None: 
            return frame, None
        
        kpts = pred.keypoints[0]
        conf = pred.keypoint_scores[0]
        vis_frame = frame.copy()
        _draw_lower_only(vis_frame, kpts, conf, kpt_thr, self.kpt_name)
        return vis_frame, {idx: (float(kpts[idx][0]), float(kpts[idx][1]), float(conf[idx])) 
                          for idx in LOWER_KPT_IDXS}

def process_video(video_path, output_dir, runner):
    """处理单个拼接视频"""
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 - {video_path}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 - {video_path}")
        return False
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_w = w // 2
    
    print(f"视频信息:")
    print(f"  路径: {video_path}")
    print(f"  尺寸: {w}x{h}")
    print(f"  FPS: {fps}")
    print(f"  总帧数: {total_frames}")
    print(f"  左右分割: {half_w}px | {half_w}px")
    
    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_path = os.path.join(output_dir, 'visualized_stereo.mp4')
    vw_stereo = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
    
    # 创建CSV文件
    csv_l_path = os.path.join(output_dir, 'keypoints_2d_left.csv')
    csv_r_path = os.path.join(output_dir, 'keypoints_2d_right.csv')
    csv_l = open(csv_l_path, 'w', newline='')
    csv_r = open(csv_r_path, 'w', newline='')
    writer_l = csv.writer(csv_l)
    writer_r = csv.writer(csv_r)
    header = ["frame", "kpt_idx", "kpt_name", "x", "y", "conf"]
    writer_l.writerow(header)
    writer_r.writerow(header)
    
    print(f"\n开始处理...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 分离左右两半
        fl = frame[:, :half_w]
        fr = frame[:, half_w:]
        
        # 分别处理
        vl, kl = runner.process_frame(fl)
        vr, kr = runner.process_frame(fr)
        
        # 拼接可视化结果
        stereo_vis = np.hstack([vl, vr])
        vw_stereo.write(stereo_vis)
        
        # 保存关键点数据
        if kl:
            for i, (x, y, c) in kl.items():
                writer_l.writerow([frame_idx, i, runner.kpt_name[i], x, y, c])
        if kr:
            for i, (x, y, c) in kr.items():
                writer_r.writerow([frame_idx, i, runner.kpt_name[i], x, y, c])
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            progress = frame_idx * 100 // total_frames
            print(f"  进度: {frame_idx}/{total_frames} ({progress}%)")
    
    cap.release()
    vw_stereo.release()
    csv_l.close()
    csv_r.close()
    
    print(f"\n处理完成!")
    print(f"  处理帧数: {frame_idx}")
    print(f"  输出视频: {out_video_path}")
    print(f"  左目CSV: {csv_l_path}")
    print(f"  右目CSV: {csv_r_path}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用Halpe8权重处理单个拼接视频')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备 (cuda:0 或 cpu)')
    parser.add_argument(
        '--standard-order',
        action='store_true',
        help='Use left-first Halpe8 labels for a checkpoint retrained with the corrected mapping. '
             'The bundled legacy checkpoint needs the default swapped order.',
    )
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 模型配置
    det_cfg = os.path.join(script_dir, "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py")
    det_ckpt = os.path.join(script_dir, "checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")
    pose_cfg = os.path.join(script_dir, "configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-lower8-256x192.py")
    pose_ckpt = os.path.join(script_dir, "checkpoints/coco_halpe_mydataset.pth")
    
    print("="*60)
    print("Halpe8 (8关键点) 拼接视频处理")
    print("="*60)
    print(f"检测器: RTMDet-m")
    print(f"姿态估计: RTMPose-m (Halpe8 自定义权重)")
    print(f"关键点: 左右髋、膝、踝、脚趾 (8个)")
    print(f"设备: {args.device}")
    print("="*60)
    
    # 初始化模型
    print("\n加载模型...")
    runner = PoseRunner(
        det_cfg,
        det_ckpt,
        pose_cfg,
        pose_ckpt,
        device=args.device,
        legacy_swapped_order=not args.standard_order,
    )
    print("模型加载完成!")
    
    # 处理视频
    print("\n" + "="*60)
    success = process_video(args.video, args.output, runner)
    print("="*60)
    
    if success:
        print("\n✓ 处理成功!")
    else:
        print("\n✗ 处理失败!")
