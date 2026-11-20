"""
双目下肢骨骼3D实时可视化
三视图（正面/侧面/俯视）纯OpenCV渲染，无额外依赖
"""
import argparse, os, sys, time
import cv2
import numpy as np

from runtime_compat import (
    CAMERA_BACKEND_CHOICES,
    DEFAULT_LEFT_CAMERA_ID,
    DEFAULT_RIGHT_CAMERA_ID,
    open_stereo_captures,
    resolve_inference_device,
)

# ── 关键点 & 骨骼定义 ─────────────────────────────────────────
LOWER_KPT_IDXS = [11, 12, 13, 14, 15, 16, 19, 20, 21]
KPT_NAME = {11:"lhip",12:"rhip",13:"lkne",14:"rkne",
            15:"lank",16:"rank",19:"hip",20:"ltoe",21:"rtoe"}
EDGES = [(11,13),(13,15),(15,20),(12,14),(14,16),(16,21),(11,12)]
# BGR颜色
C_LEFT  = (219, 142,  38)   # 蓝
C_RIGHT = ( 60,  76, 231)   # 红
C_HIP   = ( 50, 180,  50)   # 绿
EDGE_C  = {(11,13):C_LEFT,(13,15):C_LEFT,(15,20):C_LEFT,
           (12,14):C_RIGHT,(14,16):C_RIGHT,(16,21):C_RIGHT,
           (11,12):C_HIP}
KPT_C   = {11:C_LEFT,13:C_LEFT,15:C_LEFT,20:C_LEFT,
           12:C_RIGHT,14:C_RIGHT,16:C_RIGHT,21:C_RIGHT,19:C_HIP}

# ── 三视图渲染 ────────────────────────────────────────────────
VIEW_SIZE = 400   # 每个视图像素尺寸
MARGIN    = 40    # 边距

def _project(pts3d: dict, axis_x: int, axis_y: int, flip_y: bool = False):
    """把3D点投影到指定两轴平面，返回 {idx: (px, py)}"""
    coords = {}
    for idx, pt in pts3d.items():
        if pt is None: continue
        coords[idx] = (float(pt[axis_x]), float(pt[axis_y]) * (-1 if flip_y else 1))
    return coords

def _fit_scale(coords_list: list, size: int, margin: int):
    """计算统一缩放和平移，使所有点居中"""
    all_x = [v[0] for c in coords_list for v in c.values()]
    all_y = [v[1] for c in coords_list for v in c.values()]
    if not all_x:
        return 1.0, size//2, size//2
    cx = (max(all_x)+min(all_x))/2
    cy = (max(all_y)+min(all_y))/2
    span = max(max(all_x)-min(all_x), max(all_y)-min(all_y), 1.0)
    scale = (size - 2*margin) / span
    ox = size//2 - int(cx * scale)
    oy = size//2 - int(cy * scale)
    return scale, ox, oy

def _draw_view(canvas, coords: dict, scale, ox, oy, title: str):
    """在canvas上画骨骼投影"""
    def to_px(x, y):
        return (int(x * scale) + ox, int(y * scale) + oy)

    # 连线
    for a, b in EDGES:
        if a not in coords or b not in coords: continue
        p1 = to_px(*coords[a])
        p2 = to_px(*coords[b])
        cv2.line(canvas, p1, p2, EDGE_C.get((a,b),(200,200,200)), 2, cv2.LINE_AA)
    # 关节点
    for idx, (x, y) in coords.items():
        px = to_px(x, y)
        c = KPT_C.get(idx, (200,200,200))
        cv2.circle(canvas, px, 5, c, -1, cv2.LINE_AA)
        cv2.putText(canvas, KPT_NAME.get(idx,''), (px[0]+6, px[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)
    # 标题
    cv2.putText(canvas, title, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
    # 原点十字
    ox_px, oy_px = to_px(0, 0)
    cv2.drawMarker(canvas, (ox_px, oy_px), (80,80,80),
                   cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)

def render_3views(pts3d: dict) -> np.ndarray:
    """生成三视图拼接图像 (正面|侧面|俯视)"""
    S = VIEW_SIZE
    # 坐标轴约定（左相机坐标系）：X=左右, Y=上下(朝下为正), Z=深度
    # 正面：X vs Y（Y朝下为正，不翻转）
    front  = _project(pts3d, 0, 1, flip_y=False)
    # 侧面：Z vs Y
    side   = _project(pts3d, 2, 1, flip_y=False)
    # 俯视：X vs Z
    top    = _project(pts3d, 0, 2, flip_y=False)

    # 统一缩放（三视图共用同一scale，保持比例一致）
    scale, ox, oy = _fit_scale([front, side, top], S, MARGIN)

    c_front = np.zeros((S, S, 3), dtype=np.uint8)
    c_side  = np.zeros((S, S, 3), dtype=np.uint8)
    c_top   = np.zeros((S, S, 3), dtype=np.uint8)

    _draw_view(c_front, front, scale, ox, oy, "正面  (X-Y)")
    _draw_view(c_side,  side,  scale, ox, oy, "侧面  (Z-Y)")
    _draw_view(c_top,   top,   scale, ox, oy, "俯视  (X-Z)")

    # 分隔线
    sep = np.full((S, 2, 3), 60, dtype=np.uint8)
    return np.hstack([c_front, sep, c_side, sep, c_top])


def _rectify_pixel(pt_xy, K, D, R, P):
    pts = np.array([[pt_xy.astype(np.float64)]], dtype=np.float64)
    P_use = P[:, :3] if P.shape == (3, 4) else P
    out = cv2.undistortPoints(pts, K, D, R=R, P=P_use)
    return out.reshape(2).astype(np.float64)

def _triangulate(P1, P2, pt1, pt2):
    X_h = cv2.triangulatePoints(P1, P2,
                                 pt1.reshape(2,1).astype(np.float64),
                                 pt2.reshape(2,1).astype(np.float64))
    w = float(X_h[3, 0])
    if abs(w) < 1e-9: return None
    return (X_h[:3, 0] / w).astype(np.float64)

def _draw_2d(img, kpts_xy, kpts_conf, thr):
    for a, b in EDGES:
        if kpts_conf[a] < thr or kpts_conf[b] < thr: continue
        p1 = (int(kpts_xy[a][0]), int(kpts_xy[a][1]))
        p2 = (int(kpts_xy[b][0]), int(kpts_xy[b][1]))
        cv2.line(img, p1, p2, (0,255,0), 1, cv2.LINE_AA)
    for idx in LOWER_KPT_IDXS:
        if kpts_conf[idx] < thr: continue
        xi, yi = int(kpts_xy[idx][0]), int(kpts_xy[idx][1])
        cv2.circle(img, (xi, yi), 3, (0,255,0), -1, cv2.LINE_AA)


# ── 姿态推理 ──────────────────────────────────────────────────
class PoseRunner:
    def __init__(self, det_config, det_ckpt, pose_config, pose_ckpt, device, bbox_thr, nms_thr, kpt_thr):
        import torch
        from mmdet.apis import inference_detector, init_detector
        from mmpose.apis import inference_topdown, init_model as init_pose_estimator
        from mmpose.evaluation.functional import nms
        from mmpose.structures import merge_data_samples
        from mmpose.utils import adapt_mmdet_pipeline

        device = resolve_inference_device(device, torch)
        self._torch = torch
        self._infer_det  = inference_detector
        self._infer_pose = inference_topdown
        self._nms   = nms
        self._merge = merge_data_samples
        self._bbox_thr = bbox_thr
        self._nms_thr  = nms_thr
        self._kpt_thr  = kpt_thr
        self.detector = init_detector(det_config, det_ckpt, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        self.pose_estimator = init_pose_estimator(pose_config, pose_ckpt, device=device)

    def infer(self, frame_bgr):
        with self._torch.no_grad():
            det = self._infer_det(self.detector, frame_bgr)
        pred = det.pred_instances.cpu().numpy()
        bboxes = np.concatenate([pred.bboxes, pred.scores[:,None]], axis=1)
        bboxes = bboxes[(pred.labels == 0) & (pred.scores > self._bbox_thr)]
        if len(bboxes) == 0: return frame_bgr, None
        keep = self._nms(bboxes, self._nms_thr)
        bboxes = bboxes[keep]
        areas = (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
        best = int(np.argmax(areas * bboxes[:,4]))
        with self._torch.no_grad():
            pose_res = self._infer_pose(self.pose_estimator, frame_bgr, bboxes[best:best+1,:4])
        data = self._merge(pose_res)
        pred_inst = data.get('pred_instances', None)
        if pred_inst is None: return frame_bgr, None
        kpts   = pred_inst.keypoints
        scores = pred_inst.keypoint_scores
        if not isinstance(kpts,   np.ndarray): kpts   = kpts.cpu().numpy()
        if not isinstance(scores, np.ndarray): scores = scores.cpu().numpy()
        k0, c0 = kpts[0], scores[0]
        vis = frame_bgr.copy()
        _draw_2d(vis, k0, c0, self._kpt_thr)
        return vis, {idx: (float(k0[idx][0]), float(k0[idx][1]), float(c0[idx]))
                     for idx in LOWER_KPT_IDXS}


# ── 主循环 ────────────────────────────────────────────────────
def run(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from calibration.CameraConfig import CameraConfig
    cfg = CameraConfig(args.camera_config)
    P1 = cfg.P1.astype(np.float64); P2 = cfg.P2.astype(np.float64)
    K1 = cfg.left_camera_matrix.astype(np.float64)
    D1 = cfg.left_distortion.astype(np.float64)
    K2 = cfg.right_camera_matrix.astype(np.float64)
    D2 = cfg.right_distortion.astype(np.float64)
    R1 = cfg.R1.astype(np.float64); R2 = cfg.R2.astype(np.float64)

    pose_runner = PoseRunner(
        args.det_config, args.det_checkpoint,
        args.pose_config, args.pose_checkpoint,
        args.device, args.bbox_thr, args.nms_thr, args.kpt_thr)

    left_cap, right_cap = open_stereo_captures(
        args.left, args.right, backend=args.backend)
    for cap in [left_cap, right_cap]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS,          args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    print("预热中...")
    for _ in range(60):
        ok_l = left_cap.grab()
        ok_r = right_cap.grab()
        if not ok_l or not ok_r:
            left_cap.release()
            right_cap.release()
            raise RuntimeError(
                f'摄像头预热抓帧失败: left={ok_l}, right={ok_r}。'
                '请确认设备编号和相机权限。')
    print("开始，按 Q 退出")

    last_kpts_l = last_kpts_r = None
    last_3d_img = np.zeros((VIEW_SIZE, VIEW_SIZE*3+4, 3), dtype=np.uint8)
    frame_idx = 0
    t0 = time.perf_counter()

    cv2.namedWindow('Stereo 2D', cv2.WINDOW_NORMAL)
    cv2.namedWindow('3D Skeleton', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stereo 2D', args.width*2, args.height)
    cv2.resizeWindow('3D Skeleton', VIEW_SIZE*3+4, VIEW_SIZE)

    try:
        while True:
            frame_idx += 1
            ok_gl = left_cap.grab()
            ok_gr = right_cap.grab()
            if not ok_gl or not ok_gr:
                raise RuntimeError(
                    f'第 {frame_idx} 帧抓取失败: left={ok_gl}, right={ok_gr}')
            ok_l, frame_l = left_cap.retrieve()
            ok_r, frame_r = right_cap.retrieve()
            if not ok_l or not ok_r:
                raise RuntimeError(
                    f'第 {frame_idx} 帧解码失败: left={ok_l}, right={ok_r}')

            if frame_idx % args.infer_interval == 0:
                frame_l_vis, last_kpts_l = pose_runner.infer(frame_l)
                frame_r_vis, last_kpts_r = pose_runner.infer(frame_r)
            else:
                # 非推理帧：在新帧上重绘上一次关键点，消除频闪
                frame_l_vis = frame_l.copy()
                frame_r_vis = frame_r.copy()
                if last_kpts_l:
                    k = np.array([[v[0], v[1]] for v in [last_kpts_l.get(i,(0,0,0)) for i in range(max(LOWER_KPT_IDXS)+1)]])
                    c = np.array([last_kpts_l.get(i,(0,0,0))[2] for i in range(max(LOWER_KPT_IDXS)+1)])
                    _draw_2d(frame_l_vis, k, c, args.kpt_thr)
                if last_kpts_r:
                    k = np.array([[v[0], v[1]] for v in [last_kpts_r.get(i,(0,0,0)) for i in range(max(LOWER_KPT_IDXS)+1)]])
                    c = np.array([last_kpts_r.get(i,(0,0,0))[2] for i in range(max(LOWER_KPT_IDXS)+1)])
                    _draw_2d(frame_r_vis, k, c, args.kpt_thr)

            # 三角化
            if last_kpts_l and last_kpts_r:
                pts3d = {}
                for idx in LOWER_KPT_IDXS:
                    vL = last_kpts_l.get(idx); vR = last_kpts_r.get(idx)
                    if not vL or not vR: pts3d[idx] = None; continue
                    xL,yL,cL = vL; xR,yR,cR = vR
                    if cL < args.kpt_thr or cR < args.kpt_thr: pts3d[idx] = None; continue
                    ptLr = _rectify_pixel(np.array([xL,yL]), K1, D1, R1, P1)
                    ptRr = _rectify_pixel(np.array([xR,yR]), K2, D2, R2, P2)
                    pts3d[idx] = _triangulate(P1, P2, ptLr, ptRr)
                last_3d_img = render_3views(pts3d)

            fps_now = frame_idx / (time.perf_counter() - t0)
            cv2.putText(frame_l_vis, f"L {fps_now:.1f}fps", (8,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
            cv2.putText(frame_r_vis, f"R {fps_now:.1f}fps", (8,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)

            cv2.imshow('Stereo 2D', np.hstack([frame_l_vis, frame_r_vis]))
            cv2.imshow('3D Skeleton', last_3d_img)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break
    finally:
        left_cap.release(); right_cap.release()
        cv2.destroyAllWindows()


def main():
    d = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--left',   type=int, default=DEFAULT_LEFT_CAMERA_ID)
    p.add_argument('--right',  type=int, default=DEFAULT_RIGHT_CAMERA_ID)
    p.add_argument('--width',  type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps',    type=int, default=30)
    p.add_argument('--infer-interval', type=int, default=2)
    p.add_argument('--backend', choices=CAMERA_BACKEND_CHOICES, default='auto')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--bbox-thr',  type=float, default=0.5)
    p.add_argument('--nms-thr',   type=float, default=0.3)
    p.add_argument('--kpt-thr',   type=float, default=0.3)
    p.add_argument('--camera-config', default=os.path.join(d,'configs','camera_config_dechuang_640x480_LZ.yaml'))
    p.add_argument('--det-config',    default=os.path.join(d,'demo','mmdetection_cfg','rtmdet_tiny_8xb32-300e_coco.py'))
    p.add_argument('--det-checkpoint',default=os.path.join(d,'checkpoints','rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'))
    p.add_argument('--pose-config',   default=os.path.join(d,'configs','body_2d_keypoint','rtmpose','body8','rtmpose-t_8xb256-420e_body8-halpe26-256x192.py'))
    p.add_argument('--pose-checkpoint',default=os.path.join(d,'checkpoints','rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth'))
    args = p.parse_args()
    run(args)

if __name__ == '__main__':
    main()
