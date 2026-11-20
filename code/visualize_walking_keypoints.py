"""Create an interactive lower-limb walking animation from the user's CSVs.

The default session is one of the 12 non-overlap recordings.  If compensated
CSV files do not yet exist, the script creates them first, then embeds the
actual left/right 2-D trajectories and estimated 3-D skeleton into one local
HTML file.  No network connection or extra Python package is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

from stereo_occlusion_compensation import (
    JOINT_ORDER,
    StereoCalibration,
    run_csv_compensation,
)


JOINT_NAMES_CN = {
    "lhip": "左髋",
    "rhip": "右髋",
    "hip": "髋中心",
    "lkne": "左膝",
    "rkne": "右膝",
    "lank": "左踝",
    "rank": "右踝",
    "ltoe": "左脚尖",
    "rtoe": "右脚尖",
}


def _read_by_frame(path: Path) -> Dict[int, Dict[str, Dict[str, str]]]:
    result: Dict[int, Dict[str, Dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            result.setdefault(frame, {})[row["kpt_name"]] = row
    return result


def _number(row: Mapping[str, str], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return round(value, 2) if math.isfinite(value) else None


def _finite_number(row: Mapping[str, str], key: str) -> float | None:
    """Read a finite value without rounding it before geometric calculations."""

    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _confidence(row: Mapping[str, str], key: str) -> float | None:
    value = _finite_number(row, key)
    return None if value is None else round(value, 6)


def _build_html(
    session: str,
    left_path: Path,
    right_path: Path,
    three_d_path: Path,
    raw_left_path: Path,
    raw_right_path: Path,
    camera_config: Path,
    output_path: Path,
    stride: int,
    source_fps: float,
    legacy_output_path: Path | None = None,
    comparison_output_path: Path | None = None,
    raw_three_d_output_path: Path | None = None,
) -> tuple[int, int, int, int]:
    left = _read_by_frame(left_path)
    right = _read_by_frame(right_path)
    three_d = _read_by_frame(three_d_path)
    raw_left = _read_by_frame(raw_left_path)
    raw_right = _read_by_frame(raw_right_path)
    calibration = StereoCalibration.from_yaml(camera_config)
    shared_frames = sorted(set(left) & set(right) & set(three_d))[::stride]
    if not shared_frames:
        raise ValueError("左右2D与3D输出没有共同帧")

    left_compensated = 0
    right_compensated = 0
    raw_three_d_count = 0
    raw_three_d_rows: list[dict[str, object]] = []
    compact_frames = []
    for frame in shared_frames:
        left_points = []
        right_points = []
        points_3d = []
        for joint in JOINT_ORDER:
            left_row = left[frame].get(joint)
            right_row = right[frame].get(joint)
            row_3d = three_d[frame].get(joint)
            raw_left_row = raw_left.get(frame, {}).get(joint)
            raw_right_row = raw_right.get(frame, {}).get(joint)
            if left_row is None or right_row is None or row_3d is None:
                left_points.append(None)
                right_points.append(None)
                points_3d.append(None)
                continue
            left_is_compensated = left_row.get("origin") == "compensated"
            right_is_compensated = right_row.get("origin") == "compensated"
            left_compensated += int(left_is_compensated)
            right_compensated += int(right_is_compensated)
            reprojection_error = None
            raw_point_3d = None
            raw_left_conf = _finite_number(raw_left_row or {}, "conf")
            raw_right_conf = _finite_number(raw_right_row or {}, "conf")
            left_raw_xy: tuple[float | None, float | None] = (None, None)
            right_raw_xy: tuple[float | None, float | None] = (None, None)
            if raw_left_row is not None and raw_right_row is not None:
                left_raw_xy = (
                    _finite_number(raw_left_row, "x"),
                    _finite_number(raw_left_row, "y"),
                )
                right_raw_xy = (
                    _finite_number(raw_right_row, "x"),
                    _finite_number(raw_right_row, "y"),
                )
                if None not in left_raw_xy and None not in right_raw_xy:
                    point = calibration.triangulate(left_raw_xy, right_raw_xy)
                    if point is not None:
                        value = calibration.reprojection_error(point, left_raw_xy, right_raw_xy)
                        reprojection_error = round(value, 2) if math.isfinite(value) else None
                        if (
                            raw_left_conf is not None
                            and raw_right_conf is not None
                            and raw_left_conf >= 0.3
                            and raw_right_conf >= 0.3
                        ):
                            raw_point_3d = point
                            raw_three_d_count += 1
            if raw_left_row is None:
                raw_reason = "missing_left_row"
            elif raw_right_row is None:
                raw_reason = "missing_right_row"
            elif None in left_raw_xy:
                raw_reason = "invalid_left_xy"
            elif None in right_raw_xy:
                raw_reason = "invalid_right_xy"
            elif raw_left_conf is None or raw_left_conf < 0.3:
                raw_reason = "left_conf_below_0.3"
            elif raw_right_conf is None or raw_right_conf < 0.3:
                raw_reason = "right_conf_below_0.3"
            elif raw_point_3d is None:
                raw_reason = "invalid_stereo_geometry"
            else:
                raw_reason = ""
            raw_three_d_rows.append(
                {
                    "frame": frame,
                    "kpt_idx": left_row.get("kpt_idx", ""),
                    "kpt_name": joint,
                    "X": "" if raw_point_3d is None else float(raw_point_3d[0]),
                    "Y": "" if raw_point_3d is None else float(raw_point_3d[1]),
                    "Z": "" if raw_point_3d is None else float(raw_point_3d[2]),
                    "conf_l": "" if raw_left_conf is None else raw_left_conf,
                    "conf_r": "" if raw_right_conf is None else raw_right_conf,
                    "reprojection_error_px": (
                        "" if reprojection_error is None else reprojection_error
                    ),
                    "status": "raw_stereo" if raw_point_3d is not None else "unresolved",
                    "source_view": "both" if raw_point_3d is not None else "none",
                    "reason": raw_reason,
                }
            )
            # Compact 2-D layout used by the standalone page:
            # final x/y, compensation flag, original/support confidence,
            # method status/source, raw x/y/confidence, stereo reprojection error.
            left_points.append(
                [
                    _number(left_row, "x"),
                    _number(left_row, "y"),
                    int(left_is_compensated),
                    _confidence(left_row, "original_conf"),
                    _confidence(left_row, "support_conf"),
                    left_row.get("status", ""),
                    left_row.get("source_view", ""),
                    _number(raw_left_row or {}, "x"),
                    _number(raw_left_row or {}, "y"),
                    _confidence(raw_left_row or {}, "conf"),
                    reprojection_error,
                ]
            )
            right_points.append(
                [
                    _number(right_row, "x"),
                    _number(right_row, "y"),
                    int(right_is_compensated),
                    _confidence(right_row, "original_conf"),
                    _confidence(right_row, "support_conf"),
                    right_row.get("status", ""),
                    right_row.get("source_view", ""),
                    _number(raw_right_row or {}, "x"),
                    _number(raw_right_row or {}, "y"),
                    _confidence(raw_right_row or {}, "conf"),
                    reprojection_error,
                ]
            )
            # Compact 3-D layout: compensated X/Y/Z, compensation flag, status,
            # two-view confidence, source view, original visibility flags, then
            # raw stereo-only X/Y/Z.  Raw 3-D is deliberately absent unless both
            # original 2-D detections have confidence >= 0.3.
            points_3d.append(
                [
                    _number(row_3d, "X"),
                    _number(row_3d, "Y"),
                    _number(row_3d, "Z"),
                    int(row_3d.get("status", "stereo") != "stereo"),
                    row_3d.get("status", ""),
                    _confidence(row_3d, "conf_l"),
                    _confidence(row_3d, "conf_r"),
                    row_3d.get("source_view", ""),
                    int(float(row_3d.get("left_visible", "0") or 0)),
                    int(float(row_3d.get("right_visible", "0") or 0)),
                    None if raw_point_3d is None else round(float(raw_point_3d[0]), 2),
                    None if raw_point_3d is None else round(float(raw_point_3d[1]), 2),
                    None if raw_point_3d is None else round(float(raw_point_3d[2]), 2),
                ]
            )
        compact_frames.append([frame, left_points, right_points, points_3d])

    payload = {
        "session": session,
        "jointNames": list(JOINT_ORDER),
        "jointNamesCn": [JOINT_NAMES_CN[name] for name in JOINT_ORDER],
        "fps": max(1.0, source_fps / stride),
        "stride": stride,
        "frames": compact_frames,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    # The old three-panel page needs only coordinates and the compensation
    # marker.  Keeping its payload small makes it quick to open directly from
    # Finder/PyCharm while the richer v2 page retains all diagnostics.
    legacy_payload = {
        "session": session,
        "jointNames": list(JOINT_ORDER),
        "jointNamesCn": [JOINT_NAMES_CN[name] for name in JOINT_ORDER],
        "fps": max(1.0, source_fps / stride),
        "stride": stride,
        "frames": [
            [
                frame,
                [None if point is None else point[:3] for point in left_points],
                [None if point is None else point[:3] for point in right_points],
                [None if point is None else point[:4] for point in points_3d],
            ]
            for frame, left_points, right_points, points_3d in compact_frames
        ],
    }
    legacy_data_json = json.dumps(
        legacy_payload, ensure_ascii=False, separators=(",", ":")
    )

    html_text = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>下肢关键点走路动画</title>
<style>
  :root{color-scheme:light dark;--bg:light-dark(#f5f7fa,#111827);--panel:light-dark(#fff,#1f2937);--text:light-dark(#172033,#f3f4f6);--muted:light-dark(#667085,#aab4c3);--border:light-dark(#d8dee8,#445064);--left:#2563eb;--right:#dc2626;--comp:#f59e0b;--hip:#64748b}
  *{box-sizing:border-box}body{margin:0;padding:20px;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;background:var(--bg);color:var(--text)}
  h1{margin:0 0 5px;font-size:24px}.sub{color:var(--muted);margin-bottom:16px;font-size:14px}.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}.panel{background:var(--panel);border:1px solid var(--border);padding:10px}.panel h2{font-size:15px;margin:0 0 8px}canvas{display:block;width:100%;aspect-ratio:4/3;background:light-dark(#fbfdff,#111827)}
  .controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:14px;padding:12px;background:var(--panel);border:1px solid var(--border)}button,select{font:inherit;padding:7px 12px}input[type=range]{flex:1;min-width:260px}.legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:10px;font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.status{font-variant-numeric:tabular-nums;min-width:190px}.note{font-size:13px;color:var(--muted);margin-top:10px}
  @media(max-width:1050px){.grid{grid-template-columns:1fr}.panel{max-width:720px;margin:auto;width:100%}}
</style>
<h1>下肢关键点走路动画</h1>
<div class="sub" id="subtitle"></div>
<div class="grid">
  <section class="panel"><h2>左相机二维关键点</h2><canvas id="left" width="640" height="480"></canvas></section>
  <section class="panel"><h2>右相机二维关键点</h2><canvas id="right" width="640" height="480"></canvas></section>
  <section class="panel"><h2>三维骨架估计（含单侧遮挡补偿，不是真值）</h2><canvas id="three" width="640" height="480"></canvas></section>
</div>
<div class="controls">
  <button id="play" type="button">播放</button>
  <input id="slider" type="range" min="0" value="0" step="1">
  <label>速度 <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
  <label><input id="trail" type="checkbox" checked> 显示脚踝/脚尖轨迹</label>
  <span class="status" id="status"></span>
</div>
<div class="legend">
  <span><i class="dot" style="background:var(--left)"></i>左腿</span>
  <span><i class="dot" style="background:var(--right)"></i>右腿</span>
  <span><i class="dot" style="background:var(--comp)"></i>橙色外圈：由另一相机恢复</span>
</div>
<div class="note">画面来自CSV关键点，不包含原始视频背景。左右二维是相机像素坐标；三维是双目检测/补偿得到的估计值。</div>
<script>
const DATA = __DATA__;
const edges=[[0,3],[3,5],[5,7],[1,4],[4,6],[6,8],[0,1],[2,0],[2,1]];
const leftSet=new Set([0,3,5,7]), rightSet=new Set([1,4,6,8]);
const leftColor="#2563eb",rightColor="#dc2626",hipColor="#64748b",compColor="#f59e0b",gridColor="#cbd5e1";
const canvases={left:document.getElementById("left"),right:document.getElementById("right"),three:document.getElementById("three")};
const slider=document.getElementById("slider"), play=document.getElementById("play"), speed=document.getElementById("speed"), trail=document.getElementById("trail"), status=document.getElementById("status");
slider.max=String(DATA.frames.length-1);document.getElementById("subtitle").textContent=`session ${DATA.session} · 原始帧步长 ${DATA.stride} · 共 ${DATA.frames.length} 个动画帧`;
let index=0,playing=false,lastTime=0;
function colorFor(i){return leftSet.has(i)?leftColor:rightSet.has(i)?rightColor:hipColor}
function grid(ctx){ctx.clearRect(0,0,640,480);ctx.strokeStyle=gridColor;ctx.globalAlpha=.22;ctx.lineWidth=1;for(let x=0;x<=640;x+=80){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,480);ctx.stroke()}for(let y=0;y<=480;y+=80){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(640,y);ctx.stroke()}ctx.globalAlpha=1;ctx.strokeStyle=gridColor;ctx.strokeRect(.5,.5,639,479)}
function drawTrail2d(ctx,side,pointIndex){if(!trail.checked)return;ctx.strokeStyle=colorFor(pointIndex);ctx.globalAlpha=.28;ctx.lineWidth=2;ctx.beginPath();let started=false;for(let k=Math.max(0,index-24);k<=index;k++){const p=DATA.frames[k][side][pointIndex];if(!p)continue;if(!started){ctx.moveTo(p[0],p[1]);started=true}else ctx.lineTo(p[0],p[1])}ctx.stroke();ctx.globalAlpha=1}
function draw2d(canvas,side){const ctx=canvas.getContext("2d");grid(ctx);[5,6,7,8].forEach(i=>drawTrail2d(ctx,side,i));const pts=DATA.frames[index][side];for(const [a,b] of edges){const p=pts[a],q=pts[b];if(!p||!q)continue;ctx.strokeStyle=(leftSet.has(a)&&leftSet.has(b))?leftColor:(rightSet.has(a)&&rightSet.has(b))?rightColor:hipColor;ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke()}pts.forEach((p,i)=>{if(!p)return;ctx.fillStyle=colorFor(i);ctx.beginPath();ctx.arc(p[0],p[1],6,0,Math.PI*2);ctx.fill();if(p[2]){ctx.strokeStyle=compColor;ctx.lineWidth=4;ctx.beginPath();ctx.arc(p[0],p[1],11,0,Math.PI*2);ctx.stroke()}ctx.fillStyle="#111827";ctx.font="12px sans-serif";ctx.fillText(DATA.jointNamesCn[i],p[0]+9,p[1]-8)})}
function rotate3(p,center){const yaw=-.58,pitch=.28;const x=p[0]-center[0],y=-(p[1]-center[1]),z=p[2]-center[2];const x1=x*Math.cos(yaw)-z*Math.sin(yaw),z1=x*Math.sin(yaw)+z*Math.cos(yaw);const y1=y*Math.cos(pitch)-z1*Math.sin(pitch);return[x1,y1]}
function computeThreeBounds(){let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;for(const frame of DATA.frames){const pts=frame[3],center=pts[2]||pts[0]||pts[1];if(!center)continue;for(const p of pts){if(!p||![p[0],p[1],p[2]].every(Number.isFinite))continue;const q=rotate3(p,center);x0=Math.min(x0,q[0]);x1=Math.max(x1,q[0]);y0=Math.min(y0,q[1]);y1=Math.max(y1,q[1])}}if(!Number.isFinite(x0)||!Number.isFinite(y0))return{xMid:0,yMid:0,scale:1};return{xMid:(x0+x1)/2,yMid:(y0+y1)/2,scale:Math.min(500/Math.max(1,x1-x0),350/Math.max(1,y1-y0))}}
const THREE_BOUNDS=computeThreeBounds();
function project3(p,center){const q=rotate3(p,center);return[320+(q[0]-THREE_BOUNDS.xMid)*THREE_BOUNDS.scale,240-(q[1]-THREE_BOUNDS.yMid)*THREE_BOUNDS.scale]}
function drawTrail3d(ctx,pointIndex){if(!trail.checked)return;ctx.strokeStyle=colorFor(pointIndex);ctx.globalAlpha=.25;ctx.lineWidth=2;ctx.beginPath();let started=false;for(let k=Math.max(0,index-24);k<=index;k++){const pts=DATA.frames[k][3],p=pts[pointIndex],center=pts[2]||pts[0];if(!p||!center)continue;const q=project3(p,center);if(!started){ctx.moveTo(q[0],q[1]);started=true}else ctx.lineTo(q[0],q[1])}ctx.stroke();ctx.globalAlpha=1}
function draw3d(){const ctx=canvases.three.getContext("2d");grid(ctx);const pts=DATA.frames[index][3],center=pts[2]||pts[0]||pts[1];if(!center)return;[5,6,7,8].forEach(i=>drawTrail3d(ctx,i));const proj=pts.map(p=>p?project3(p,center):null);for(const [a,b] of edges){const p=proj[a],q=proj[b];if(!p||!q)continue;ctx.strokeStyle=(leftSet.has(a)&&leftSet.has(b))?leftColor:(rightSet.has(a)&&rightSet.has(b))?rightColor:hipColor;ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke()}proj.forEach((q,i)=>{if(!q)return;ctx.fillStyle=colorFor(i);ctx.beginPath();ctx.arc(q[0],q[1],7,0,Math.PI*2);ctx.fill();if(pts[i][3]){ctx.strokeStyle=compColor;ctx.lineWidth=4;ctx.beginPath();ctx.arc(q[0],q[1],12,0,Math.PI*2);ctx.stroke()}ctx.fillStyle="#111827";ctx.font="12px sans-serif";const tx=Math.min(580,Math.max(8,q[0]+10)),ty=Math.min(468,Math.max(14,q[1]-8));ctx.fillText(DATA.jointNamesCn[i],tx,ty)})}
function draw(){draw2d(canvases.left,1);draw2d(canvases.right,2);draw3d();slider.value=String(index);const f=DATA.frames[index][0];status.textContent=`动画 ${index+1}/${DATA.frames.length} · 原始帧 ${f}`}
function tick(t){if(!playing)return;const interval=1000/(DATA.fps*Number(speed.value));if(t-lastTime>=interval){index=(index+1)%DATA.frames.length;lastTime=t;draw()}requestAnimationFrame(tick)}
play.addEventListener("click",()=>{playing=!playing;play.textContent=playing?"暂停":"播放";if(playing){lastTime=0;requestAnimationFrame(tick)}});slider.addEventListener("input",()=>{index=Number(slider.value);draw()});trail.addEventListener("change",draw);draw();
</script>
""".replace("__DATA__", legacy_data_json)
    legacy_html_text = html_text
    # Keep the data-building logic in Python and the interactive page in a
    # separate template.  This also makes browser-only features (recording,
    # jump diagnostics, direction display) easier to inspect and maintain.
    template_path = Path(__file__).with_name("walking_visualization_template.html")
    if not template_path.is_file():
        raise FileNotFoundError(f"找不到新版动画模板：{template_path}")
    html_text = template_path.read_text(encoding="utf-8").replace("__DATA__", data_json)
    comparison_html_text = None
    if comparison_output_path is not None:
        comparison_template = Path(__file__).with_name("raw_vs_compensated_template.html")
        if not comparison_template.is_file():
            raise FileNotFoundError(f"找不到原始/补偿对比模板：{comparison_template}")
        comparison_html_text = comparison_template.read_text(encoding="utf-8").replace(
            "__DATA__", data_json
        )
    if legacy_output_path is not None:
        legacy_output_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_output_path.write_text(legacy_html_text, encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    if comparison_output_path is not None and comparison_html_text is not None:
        comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_output_path.write_text(comparison_html_text, encoding="utf-8")
    if raw_three_d_output_path is not None:
        raw_three_d_output_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_three_d_output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "frame",
                    "kpt_idx",
                    "kpt_name",
                    "X",
                    "Y",
                    "Z",
                    "conf_l",
                    "conf_r",
                    "reprojection_error_px",
                    "status",
                    "source_view",
                    "reason",
                ),
            )
            writer.writeheader()
            writer.writerows(raw_three_d_rows)
    return len(shared_frames), left_compensated, right_compensated, raw_three_d_count


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate a lower-limb walking animation")
    parser.add_argument("--session", default="20260304_221628")
    parser.add_argument("--input-root", type=Path, default=script_dir / "recordings_analysis")
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=script_dir / "configs" / "camera_config_pose_640x480.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir / "experiments" / "nonoverlap_12_sessions" / "walking_visualization",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth source frame (default 1 so one-frame detector jumps are retained)",
    )
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--force-compensation", action="store_true")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Write a named HTML file; relative paths are placed under --output-root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    session_dir = args.input_root / args.session
    left_input = session_dir / "keypoints_2d_left.csv"
    right_input = session_dir / "keypoints_2d_right.csv"
    if not left_input.is_file() or not right_input.is_file():
        raise SystemExit(f"找不到session的左右CSV：{session_dir}")
    compensation_dir = args.output_root / f"{args.session}_compensated"
    left_output = compensation_dir / "keypoints_2d_left_compensated.csv"
    right_output = compensation_dir / "keypoints_2d_right_compensated.csv"
    three_d_output = compensation_dir / "keypoints_3d_compensated.csv"
    if args.force_compensation or not all(
        path.is_file() for path in (left_output, right_output, three_d_output)
    ):
        print(f"先生成完整补偿数据：{args.session}")
        run_csv_compensation(
            left_input,
            right_input,
            args.camera_config,
            compensation_dir,
        )
    if args.output_file is None:
        # Keep every earlier page for comparison.  The v3 filename makes it
        # unambiguous that this is the compact, fully fitted version.
        animation_path = args.output_root / f"walking_keypoints_{args.session}_v3.html"
    else:
        animation_path = args.output_file
        if not animation_path.is_absolute():
            animation_path = args.output_root / animation_path
    raw_three_d_output = compensation_dir / "keypoints_3d_raw_stereo.csv"
    frame_count, left_count, right_count, raw_three_d_count = _build_html(
        args.session,
        left_output,
        right_output,
        three_d_output,
        left_input,
        right_input,
        args.camera_config,
        animation_path,
        max(1, args.stride),
        args.source_fps,
        legacy_output_path=(
            args.output_root / f"walking_keypoints_{args.session}_legacy.html"
        ),
        comparison_output_path=(
            args.output_root
            / f"walking_keypoints_{args.session}_raw_vs_compensated.html"
        ),
        raw_three_d_output_path=raw_three_d_output,
    )
    print(f"session：{args.session}")
    print(f"动画帧：{frame_count}")
    print(f"动画采样后左图补偿点：{left_count}")
    print(f"动画采样后右图补偿点：{right_count}")
    print(f"原始双目三维有效点：{raw_three_d_count}")
    print(f"原始双目三维CSV：{raw_three_d_output}")
    print(
        "旧式三面板（已修复三维取景）："
        f"{args.output_root / f'walking_keypoints_{args.session}_legacy.html'}"
    )
    print(f"新版四格紧凑版（已修复三维取景）：{animation_path}")
    print(
        "原始检测与补偿结果对比："
        f"{args.output_root / f'walking_keypoints_{args.session}_raw_vs_compensated.html'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
