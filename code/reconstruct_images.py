"""
对 recordings/extracted_frames_20260305_034722/ 下三张拼接图做三维重建
三角化逻辑参考 stereo_halpe26_knee_triangulation_optimized.py
标定文件: configs/camera_config_dechuang_640x480_LZ.yaml
输出: reconstruction_output/
"""
import os, sys, io
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImage

ROOT   = os.path.dirname(os.path.abspath(__file__))
IMGDIR = os.path.join(ROOT, 'recordings', 'extracted_frames_20260305_034722')
OUTDIR = os.path.join(ROOT, 'reconstruction_output')
os.makedirs(OUTDIR, exist_ok=True)
IMAGES = ['迈右腿.png', '迈左腿.png', '站立.png']

# ── 关键点定义（与 optimized 一致）─────────────────────────
LOWER_KPT_IDXS = [11, 12, 13, 14, 15, 16, 19, 20, 21]
KPT_NAME = {11:"左髋",12:"右髋",13:"左膝",14:"右膝",
            15:"左踝",16:"右踝",19:"髋中",20:"左趾",21:"右趾"}
LOWER_EDGES = [(11,13),(13,15),(12,14),(14,16),(15,20),(16,21),(11,12),(11,19),(12,19)]
C_L, C_R, C_H = '#E8A020', '#3C4CE7', '#32B432'
EDGE_C = {(11,13):C_L,(13,15):C_L,(15,20):C_L,
          (12,14):C_R,(14,16):C_R,(16,21):C_R,
          (11,12):C_H,(11,19):C_H,(12,19):C_H}
KPT_C  = {11:C_L,13:C_L,15:C_L,20:C_L,
          12:C_R,14:C_R,16:C_R,21:C_R,19:C_H}

# ── 三角化（直接复用 optimized 的实现）──────────────────────
def _rectify_pixel(pt_xy, K, D, R, P):
    pts = np.array([[pt_xy.astype(np.float64)]], dtype=np.float64)
    P_use = P[:, :3] if isinstance(P, np.ndarray) and P.shape == (3, 4) else P
    out = cv2.undistortPoints(pts, K, D, R=R, P=P_use)
    return out.reshape(2).astype(np.float64)

def _triangulate_point(P1, P2, pt1, pt2):
    X_h = cv2.triangulatePoints(P1, P2,
                                 pt1.reshape(2,1).astype(np.float64),
                                 pt2.reshape(2,1).astype(np.float64))
    w = float(X_h[3, 0])
    if abs(w) < 1e-9:
        return None
    return (X_h[:3, 0] / w).astype(np.float64)

# ── 加载标定（与 optimized 一致，用 CameraConfig）───────────
sys.path.insert(0, ROOT)
from calibration.CameraConfig import CameraConfig

CALIB = os.path.join(ROOT, 'calibration', 'configs', 'camera_config_pose_640x480 Z .yaml')
cfg = CameraConfig(CALIB)
P1 = cfg.P1.astype(np.float64)
P2 = cfg.P2.astype(np.float64)
K1 = cfg.left_camera_matrix.astype(np.float64)
D1 = cfg.left_distortion.astype(np.float64)
K2 = cfg.right_camera_matrix.astype(np.float64)
D2 = cfg.right_distortion.astype(np.float64)
R1 = cfg.R1.astype(np.float64)
R2 = cfg.R2.astype(np.float64)
# 坐标原点设在两相机中心（与 skeleton_gui.py 一致）
origin_offset = cfg.T.reshape(3).astype(np.float64) / 2.0
print(f"基线: {abs(cfg.T.reshape(3)[0]):.1f} mm  原点偏移: {origin_offset}")

# ── 加载模型（与 optimized 一致，用 m 模型）─────────────────
import torch
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

det_cfg  = os.path.join(ROOT, 'demo', 'mmdetection_cfg', 'rtmdet_t_640-8xb32_coco-person.py')
det_ckpt = os.path.join(ROOT, 'checkpoints', 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')
pose_cfg  = os.path.join(ROOT, 'configs', 'body_2d_keypoint', 'rtmpose', 'body8',
                         'rtmpose-t_8xb256-420e_body8-halpe26-256x192.py')
pose_ckpt = os.path.join(ROOT, 'checkpoints',
                         'rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth')

print("加载检测模型...")
detector = init_detector(det_cfg, det_ckpt, device=DEVICE)
detector.cfg = adapt_mmdet_pipeline(detector.cfg)
print("加载姿态模型...")
pose_estimator = init_pose_estimator(pose_cfg, pose_ckpt, device=DEVICE)
print("模型加载完成\n")

# ── 推理（与 optimized 的 visualize_and_get_lower_kpts 对齐）
def infer(frame_bgr, bbox_thr=0.5, nms_thr=0.3, kpt_thr=0.3):
    """返回 {idx: (x, y, conf)} 或 None
    先尝试 bbox 检测，若无结果则用全图 bbox 兜底"""
    h, w = frame_bgr.shape[:2]

    def _run_pose(bbox_xyxy):
        with torch.no_grad():
            res = inference_topdown(pose_estimator, frame_bgr,
                                    np.array([bbox_xyxy], dtype=np.float32))
        data = merge_data_samples(res)
        pi = data.get('pred_instances', None)
        if pi is None:
            return None
        k = pi.keypoints; s = pi.keypoint_scores
        if not isinstance(k, np.ndarray): k = k.cpu().numpy()
        if not isinstance(s, np.ndarray): s = s.cpu().numpy()
        k0, s0 = k[0], s[0]
        return {i: (float(k0[i][0]), float(k0[i][1]), float(s0[i]))
                for i in LOWER_KPT_IDXS}

    # 先跑检测，bbox_thr=0.3 确保主体能被检测到
    with torch.no_grad():
        det_result = inference_detector(detector, frame_bgr)
    pred = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate([pred.bboxes, pred.scores[:,None]], axis=1)
    bboxes = bboxes[(pred.labels == 0) & (pred.scores > bbox_thr)]

    if len(bboxes) > 0:
        keep = nms(bboxes, nms_thr)
        bboxes = bboxes[keep]
        # 选面积最大的 bbox（主体人物）
        areas = (bboxes[:,2]-bboxes[:,0]) * (bboxes[:,3]-bboxes[:,1])
        best = int(np.argmax(areas))
        result = _run_pose(bboxes[best, :4])
        if result is not None:
            return result

    # 兜底：用全图 bbox
    result = _run_pose([0, 0, w, h])
    return result

# ── 三维重建（与 optimized run_stereo_capture 的 3D 块对齐）─
def reconstruct_3d(kpts_l, kpts_r, kpt_thr=0.3):
    pts3d = {}
    for idx in LOWER_KPT_IDXS:
        vL = kpts_l.get(idx); vR = kpts_r.get(idx)
        if not vL or not vR or vL[2] < kpt_thr or vR[2] < kpt_thr:
            pts3d[idx] = None; continue
        ptL = np.array([vL[0], vL[1]], dtype=np.float64)
        ptR = np.array([vR[0], vR[1]], dtype=np.float64)
        ptLr = _rectify_pixel(ptL, K1, D1, R1, P1)
        ptRr = _rectify_pixel(ptR, K2, D2, R2, P2)
        X = _triangulate_point(P1, P2, ptLr, ptRr)
        pts3d[idx] = (X - origin_offset) if X is not None else None
    return pts3d

# ── 绘图 ──────────────────────────────────────────────────────
def draw_2d(ax, img_bgr, kpts, title, kpt_thr=0.3):
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=10, color='white')
    ax.axis('off')
    if not kpts: return
    for a, b in LOWER_EDGES:
        va, vb = kpts.get(a), kpts.get(b)
        if not va or not vb or va[2] < kpt_thr or vb[2] < kpt_thr: continue
        ax.plot([va[0],vb[0]], [va[1],vb[1]], color=EDGE_C.get((a,b),'#aaa'), lw=1.5)
    for idx, v in kpts.items():
        if v[2] < kpt_thr: continue
        c = KPT_C.get(idx, '#aaa')
        ax.plot(v[0], v[1], 'o', color=c, ms=5)
        ax.text(v[0]+4, v[1]-4, KPT_NAME.get(idx,''), color=c, fontsize=6)

def draw_3d_view(ax, pts3d, xlabel, ylabel, xi, yi, flip_y, title):
    ax.set_facecolor('#111111')
    ax.set_title(title, color='white', fontsize=10)
    ax.tick_params(colors='#888'); ax.spines[:].set_color('#444')
    ax.set_xlabel(xlabel, color='#888', fontsize=8)
    ax.set_ylabel(ylabel, color='#888', fontsize=8)
    coords = {}
    for idx, pt in pts3d.items():
        if pt is None: continue
        coords[idx] = (float(pt[xi]), float(pt[yi]) * (-1 if flip_y else 1))
    for a, b in LOWER_EDGES:
        if a not in coords or b not in coords: continue
        ax.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]],
                color=EDGE_C.get((a,b), '#aaa'), lw=2)
    for idx, (x, y) in coords.items():
        c = KPT_C.get(idx, '#aaa')
        ax.plot(x, y, 'o', color=c, ms=6)
        ax.text(x+2, y-2, KPT_NAME.get(idx,''), color=c, fontsize=6)
    ax.axhline(0, color='#333', lw=0.5, ls='--')
    ax.axvline(0, color='#333', lw=0.5, ls='--')

# ── 主流程 ────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

KPT_THR = 0.2   # 适当降低，抢救低置信度关键点

for fname in IMAGES:
    fpath = os.path.join(IMGDIR, fname)
    print(f"处理: {fname}")
    buf = np.fromfile(fpath, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        print("  读取失败，跳过"); continue

    mid = img.shape[1] // 2
    # 这批拼接图：左半=右相机，右半=左相机，对调
    fr = cv2.resize(img[:, :mid], (640, 480))
    fl = cv2.resize(img[:, mid:], (640, 480))

    kpts_l = infer(fl, bbox_thr=0.3, kpt_thr=KPT_THR)
    kpts_r = infer(fr, bbox_thr=0.3, kpt_thr=KPT_THR)
    print(f"  左图: {'检测到' if kpts_l else '未检测到'}  右图: {'检测到' if kpts_r else '未检测到'}")

    pts3d = {}
    if kpts_l and kpts_r:
        pts3d = reconstruct_3d(kpts_l, kpts_r, kpt_thr=KPT_THR)
        valid = sum(1 for v in pts3d.values() if v is not None)
        print(f"  三维重建: {valid}/{len(LOWER_KPT_IDXS)} 个关键点")
        for idx, pt in pts3d.items():
            if pt is not None:
                print(f"    {KPT_NAME[idx]:3s}: X={pt[0]:7.1f}  Y={pt[1]:7.1f}  Z={pt[2]:7.1f} mm")

    # ── 出图：左2D | 右2D | 正面 | 侧面 | 俯视 ──────────────
    fig = plt.figure(figsize=(20, 5), facecolor='#1e1e1e')
    fig.suptitle(f'三维重建结果 — {os.path.splitext(fname)[0]}',
                 color='white', fontsize=13, y=1.01)

    ax1 = fig.add_subplot(1,5,1); ax2 = fig.add_subplot(1,5,2)
    draw_2d(ax1, fl, kpts_l, '左相机 2D', KPT_THR)
    draw_2d(ax2, fr, kpts_r, '右相机 2D', KPT_THR)

    ax3 = fig.add_subplot(1,5,3)
    ax4 = fig.add_subplot(1,5,4)
    ax5 = fig.add_subplot(1,5,5)
    if pts3d and any(v is not None for v in pts3d.values()):
        draw_3d_view(ax3, pts3d, 'X (mm)', 'Y (mm)', 0, 1, False, '正面视图 (X-Y)')
        draw_3d_view(ax4, pts3d, 'Z (mm)', 'Y (mm)', 2, 1, False, '侧面视图 (Z-Y)')
        draw_3d_view(ax5, pts3d, 'X (mm)', 'Z (mm)', 0, 2, False, '俯视图 (X-Z)')
    else:
        for ax in [ax3, ax4, ax5]:
            ax.set_facecolor('#111')
            ax.text(0.5, 0.5, '无重建结果', color='#888',
                    ha='center', va='center', transform=ax.transAxes)

    patches = [mpatches.Patch(color=C_L, label='左侧'),
               mpatches.Patch(color=C_R, label='右侧'),
               mpatches.Patch(color=C_H, label='髋部')]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               facecolor='#1e1e1e', labelcolor='white', fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()

    out_path = os.path.join(OUTDIR, os.path.splitext(fname)[0] + '_3d重建.png')
    buf_io = io.BytesIO()
    fig.savefig(buf_io, format='png', dpi=120, bbox_inches='tight', facecolor='#1e1e1e')
    buf_io.seek(0)
    PILImage.open(buf_io).save(out_path)
    plt.close(fig)
    print(f"  已保存: {out_path}\n")

print("全部完成，结果在 reconstruction_output/")
