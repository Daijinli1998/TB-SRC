"""Open a generated walking visualization through a local-only web server.

Some file previewers render the HTML shell but block its JavaScript, leaving
the canvases empty.  Running this script from PyCharm serves the same files on
127.0.0.1 and opens the selected page in the normal browser.  No third-party
package or terminal command is required.
"""

from __future__ import annotations

import argparse
import functools
import urllib.error
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence


SESSION = "20260206_221436"
PORT = 8765


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


def _page_path(project_root: Path, version: str) -> Path:
    suffixes = {
        "new": "v3",
        "old": "legacy",
        "compare": "raw_vs_compensated",
    }
    suffix = suffixes[version]
    return (
        project_root
        / "run_halpe26_cam"
        / "experiments"
        / "nonoverlap_12_sessions"
        / "walking_visualization"
        / f"walking_keypoints_{SESSION}_{suffix}.html"
    )


def _page_url(project_root: Path, page_path: Path) -> str:
    relative = page_path.relative_to(project_root).as_posix()
    return f"http://127.0.0.1:{PORT}/{relative}"


def _already_served(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open the walking animation")
    parser.add_argument(
        "--version", choices=("new", "old", "compare"), default="new"
    )
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    page_path = _page_path(project_root, args.version)
    if not page_path.is_file():
        raise SystemExit(
            f"找不到：{page_path}\n"
            "请先在PyCharm运行“08 生成新旧两版下肢动画”。"
        )
    url = _page_url(project_root, page_path)
    if args.check_only:
        print(f"文件：{page_path}")
        print(f"地址：{url}")
        return 0

    if _already_served(url):
        webbrowser.open(url)
        print(f"已打开：{url}")
        return 0

    handler = functools.partial(_QuietHandler, directory=str(project_root))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    except OSError as error:
        raise SystemExit(
            f"本机端口{PORT}已被其他程序占用：{error}\n"
            "可以双击 walking_visualization 文件夹里的 .command 文件重试。"
        ) from error

    webbrowser.open(url)
    print(f"已打开：{url}")
    print("此运行窗口用于显示动画；看完后点击PyCharm红色停止按钮即可。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
