"""
对单张拼接图做三维重建并可视化
用法:
  python reconstruct_single_image.py <stereo_image_path> [--output <out_path>]
拼接约定: 左半 = 左相机, 右半 = 右相机
"""
import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from calibration.CameraConfig import CameraConfig

# ── 关键点定义 ──────────────────────────────────────────────
LOWER_KPT_IDXS = [11, 12, 13, 14, 15, 16, 19, 20, 21]
KPT_NAME = {
    11: "lhip", 12: "rhip", 13: "lkne", 14: "rkne",
    15: "lank", 16: "rank", 19: "hip",  20: "ltoe", 21: "rtoe",
}
LOWER_EDGES = [(11,13),(13,15),(12,14),(14,16),(15,20),(16,21),(11,19),(12,19)]

C_L = '#4A90E2'   # 蓝 - 左腿
C_R = '#E25454'   # 红 - 右腿
C_H = '#32B432'   # 绿 - hip

EDGE_C = {
    (11,13):C_L,(13,15):C_L,(15,20):C_L,
    (12,14):C_R,(14,16):C_R,(16,21):C_R,
    (11,19):C_H,(12,19):C_H,
}
KPT_C = {
    11:C_L,13:C_L,15:C_L,20:C_L,
    12:C_R,14:C_R,16:C_R,21:C_R,
    19:C_H,
}


# ── 三角化 ──────────────────────────────────────────────────
def _rectify_pixel(pt_xy, K, D, R, P):
    pts = np.array([[pt_xy.astype(np.float64)]], dtype=np.float64)
    P_use = P[:, :3] if isinstance(P, np.ndarray) and P.shape == (3, 4) else P
    out = cv2.undistortPoints(pts, K, D, R=R, P=P_use)
    return out.reshape(2).astype(np.float64)

def _triangulate_point(P1, P2, pt1, pt2):
    X_h = cv2.triangulatePoints(
        P1, P2,
        pt1.reshape(2, 1).astype(np.float64),
        pt2.reshape(2, 1).astype(np.float64),
    )
    w = float(X_h[3, 0])
    if abs(w) < 1e-9:
        return None
    return (X_h[:3, 0] / w).astype(np.float64)


# ── 推理 ────────────────────────────────────────────────────
def infer(detector, pose_estimator, frame_bgr, bbox_thr=0.3, nms_thr=0.3, kpt_thr=0.3):
    import torch
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown
    from mmpose.evaluation.functional import nms as mmpose_nms
    from mmpose.structures import merge_data_samples

    h, w = frame_bgr.shape[:2]

    def _run_pose(bbox_xyxy):
        with torch.no_grad():
            res = inference_topdown(pose_estimator, frame_bgr,
                                    np.array([bbox_xyxy], dtype=np.float32))
        data = merge_data_samples(res)
        pi = data.get('pred_instances', None)
        if pi is None:
            return None
        k = pi.keypoints
        s = pi.keypoint_scores
        if not isinstance(k, np.ndarray):
            k = k.cpu().numpy()
        if not isinstance(s, np.ndarray):
            s = s.cpu().numpy()
        k0, s0 = k[0], s[0]
        return {i: (float(k0[i][0]), float(k0[i][1]), float(s0[i]))
                for i in LOWER_KPT_IDXS}

    with torch.no_grad():
        det_result = inference_detector(detector, frame_bgr)
    pred = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate([pred.bboxes, pred.scores[:, None]], axis=1)
    bboxes = bboxes[(pred.labels == 0) & (pred.scores > bbox_thr)]

    if len(bboxes) > 0:
        keep = mmpose_nms(bboxes, nms_thr)
        bboxes = bboxes[keep]
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        best = int(np.argmax(areas))
        result = _run_pose(bboxes[best, :4])
        if result is not None:
            return result

    return _run_pose([0, 0, w, h])


# ── 绘图辅助 ────────────────────────────────────────────────
def draw_2d(ax, img_bgr, kpts, title, kpt_thr=0.3):
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    if not kpts:
        return
    for a, b in LOWER_EDGES:
        va, vb = kpts.get(a), kpts.get(b)
        if not va or not vb or va[2] < kpt_thr or vb[2] < kpt_thr:
            continue
        ax.plot([va[0], vb[0]], [va[1], vb[1]],
                color=EDGE_C.get((a, b), '#aaa'), lw=1.5)
    for idx, v in kpts.items():
        if v[2] < kpt_thr:
            continue
        c = KPT_C.get(idx, '#aaa')
        ax.plot(v[0], v[1], 'o', color=c, ms=4)
        ax.text(v[0] + 4, v[1] + 8, KPT_NAME.get(idx, ''), color=c, fontsize=6)


def draw_3d_skeleton(ax, pts3d):
    """
    全屏大3D骨架图，风格类似 batch_visualization_results 里的 01_3d_trajectory_overview.png：
      - 坐标系：X右, Y上（翻转图像Y）, Z深度（原始Z，朝相机为正）
      - 视角：elev=20, azim=45
    """
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.tick_params(labelsize=8)

    ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Z (mm)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Y (mm)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('3D Skeleton Reconstruction', fontsize=16, fontweight='bold', pad=20)

    ax.view_init(elev=20, azim=45)

    coords = {}
    for idx, pt in pts3d.items():
        if pt is None:
            continue
        x =  float(pt[0])   # X: 水平
        y =  float(pt[2])   # Z: 深度（Y轴显示）
        z = -float(pt[1])   # -Y: 翻转图像Y使向上为正（Z轴显示）
        coords[idx] = (x, y, z)

    for a, b in LOWER_EDGES:
        if a not in coords or b not in coords:
            continue
        xa, ya, za = coords[a]
        xb, yb, zb = coords[b]
        ax.plot([xa, xb], [ya, yb], [za, zb],
                color=EDGE_C.get((a, b), '#aaa'), lw=3, alpha=0.9,
                solid_capstyle='round')

    for idx, (x, y, z) in coords.items():
        c = KPT_C.get(idx, '#aaa')
        ax.scatter(x, y, z, color=c, s=100, zorder=5, depthshade=True,
                   edgecolors='black', linewidths=0.5)
        ax.text(x + 3, y + 3, z + 3, KPT_NAME.get(idx, ''), color=c, fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="拼接图路径")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--kpt-thr", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"图像不存在: {args.image}")

    if args.output is None:
        base = os.path.splitext(args.image)[0]
        args.output = base + "_3d.png"

    # 加载标定
    calib_path = os.path.join(ROOT, "calibration", "configs",
                              "camera_config_pose_640x480 Z .yaml")
    cfg = CameraConfig(calib_path)
    P1 = cfg.P1.astype(np.float64)
    P2 = cfg.P2.astype(np.float64)
    K1 = cfg.left_camera_matrix.astype(np.float64)
    D1 = cfg.left_distortion.astype(np.float64)
    K2 = cfg.right_camera_matrix.astype(np.float64)
    D2 = cfg.right_distortion.astype(np.float64)
    R1 = cfg.R1.astype(np.float64)
    R2 = cfg.R2.astype(np.float64)
    origin_offset = cfg.T.reshape(3).astype(np.float64) / 2.0

    # 加载模型
    import torch
    from mmdet.apis import init_detector
    from mmpose.apis import init_model as init_pose_estimator
    from mmpose.utils import adapt_mmdet_pipeline

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"设备: {device}")

    det_cfg  = os.path.join(ROOT, "demo", "mmdetection_cfg", "rtmdet_m_640-8xb32_coco-person.py")
    det_ckpt = os.path.join(ROOT, "checkpoints", "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")
    pose_cfg  = os.path.join(ROOT, "configs", "body_2d_keypoint", "rtmpose", "body8",
                             "rtmpose-m_8xb512-700e_body8-halpe26-256x192.py")
    pose_ckpt = os.path.join(ROOT, "checkpoints",
                             "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth")

    print("加载模型...")
    detector = init_detector(det_cfg, det_ckpt, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(pose_cfg, pose_ckpt, device=device)

    # 读图 split
    buf = np.fromfile(args.image, dtype=np.uint8)
    stereo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if stereo is None:
        raise RuntimeError(f"无法读取: {args.image}")

    h, full_w = stereo.shape[:2]
    half_w = full_w // 2
    frame_l = stereo[:, :half_w].copy()
    frame_r = stereo[:, half_w:].copy()
    print(f"图像: {full_w}x{h}, 每目: {half_w}x{h}")

    # 推理
    print("推理左目...")
    kpts_l = infer(detector, pose_estimator, frame_l, kpt_thr=args.kpt_thr)
    print("推理右目...")
    kpts_r = infer(detector, pose_estimator, frame_r, kpt_thr=args.kpt_thr)

    # 三维重建
    pts3d = {}
    if kpts_l and kpts_r:
        for idx in LOWER_KPT_IDXS:
            vL = kpts_l.get(idx)
            vR = kpts_r.get(idx)
            if not vL or not vR or vL[2] < args.kpt_thr or vR[2] < args.kpt_thr:
                pts3d[idx] = None
                continue
            ptLr = _rectify_pixel(np.array([vL[0], vL[1]]), K1, D1, R1, P1)
            ptRr = _rectify_pixel(np.array([vR[0], vR[1]]), K2, D2, R2, P2)
            X = _triangulate_point(P1, P2, ptLr, ptRr)
            if X is None:
                pts3d[idx] = None
                continue
            pt = X - origin_offset
            # 合理性过滤：Z深度 200~5000mm，XY偏移不超过2000mm
            if not (200 < pt[2] < 5000 and abs(pt[0]) < 2000 and abs(pt[1]) < 2000):
                print(f"  {KPT_NAME[idx]:5s}: 三角化异常，丢弃 X={pt[0]:.0f} Y={pt[1]:.0f} Z={pt[2]:.0f}")
                pts3d[idx] = None
                continue
            pts3d[idx] = pt

        valid = sum(1 for v in pts3d.values() if v is not None)
        print(f"三维重建: {valid}/{len(LOWER_KPT_IDXS)} 个关键点")
        for idx, pt in pts3d.items():
            if pt is not None:
                print(f"  {KPT_NAME[idx]:5s}: X={pt[0]:7.1f}  Y={pt[1]:7.1f}  Z={pt[2]:7.1f} mm")

    # 打印2D坐标
    print("\n左目2D关键点:")
    for idx in LOWER_KPT_IDXS:
        v = kpts_l.get(idx) if kpts_l else None
        print(f"  {KPT_NAME[idx]:5s}: {'x=%-7.1f y=%-7.1f conf=%.3f' % (v[0],v[1],v[2]) if v else 'N/A'}")
    print("右目2D关键点:")
    for idx in LOWER_KPT_IDXS:
        v = kpts_r.get(idx) if kpts_r else None
        print(f"  {KPT_NAME[idx]:5s}: {'x=%-7.1f y=%-7.1f conf=%.3f' % (v[0],v[1],v[2]) if v else 'N/A'}")

    # 输出2D坐标CSV
    csv_path = os.path.splitext(args.output)[0] + "_2d.csv"
    rows = []
    for idx in LOWER_KPT_IDXS:
        name = KPT_NAME.get(idx, str(idx))
        vL = kpts_l.get(idx) if kpts_l else None
        vR = kpts_r.get(idx) if kpts_r else None
        rows.append({
            'kpt': name,
            'left_x':  round(vL[0], 2) if vL else '',
            'left_y':  round(vL[1], 2) if vL else '',
            'left_conf': round(vL[2], 4) if vL else '',
            'right_x': round(vR[0], 2) if vR else '',
            'right_y': round(vR[1], 2) if vR else '',
            'right_conf': round(vR[2], 4) if vR else '',
        })
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['kpt','left_x','left_y','left_conf','right_x','right_y','right_conf'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"2D坐标已保存: {csv_path}")

    # 出图: 大3D图（主体）+ 左右2D小缩略图
    fig = plt.figure(figsize=(14, 10))

    # 大3D图占左侧约70%
    ax3 = fig.add_axes([0.0, 0.12, 0.68, 0.82], projection='3d')
    has_3d = pts3d and any(v is not None for v in pts3d.values())
    if has_3d:
        draw_3d_skeleton(ax3, pts3d)
    else:
        ax3.text(0, 0, 0, 'no 3D result', color='#888', fontsize=12)

    # 右侧两个小2D图
    ax1 = fig.add_axes([0.70, 0.55, 0.28, 0.38])
    ax2 = fig.add_axes([0.70, 0.10, 0.28, 0.38])
    draw_2d(ax1, frame_l, kpts_l, 'left cam', args.kpt_thr)
    draw_2d(ax2, frame_r, kpts_r, 'right cam', args.kpt_thr)

    # 图例
    patches = [
        mpatches.Patch(color=C_L, label='left leg'),
        mpatches.Patch(color=C_R, label='right leg'),
        mpatches.Patch(color=C_H, label='hip'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=10, bbox_to_anchor=(0.35, 0.01))

    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
