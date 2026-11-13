"""Phase 0b: 从 hf-mirror 下载模型（断点续传、跳过已完整文件、绕开系统代理）。

用法：
  python scripts/04_download_models.py              # 按优先级下全部
  python scripts/04_download_models.py --only core  # 只下阶段 1.5 必需的核心模型
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MIRROR = "https://hf-mirror.com"

TASKS = [
    # (优先级组, repo, 仓库内路径, 本地相对路径, 期望GB)
    ("core", "OnomaAIResearch/Illustrious-xl-early-release-v0", "Illustrious-XL-v0.1.safetensors",
     "checkpoints/illustriousXL_v01.safetensors", 6.46),
    ("core", "h94/IP-Adapter", "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
     "ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors", 0.79),
    ("core", "h94/IP-Adapter", "models/image_encoder/model.safetensors",
     "clip_vision/CLIP-ViT-H-14.safetensors", 2.35),
    ("core", "xinsir/controlnet-openpose-sdxl-1.0", "diffusion_pytorch_model.safetensors",
     "controlnet/controlnet-openpose-sdxl.safetensors", 2.33),
    ("core", "Kim2091/AnimeSharp", "4x-AnimeSharp.pth",
     "upscale_models/4x-AnimeSharp.pth", 0.06),
    ("extra", "xinsir/controlnet-depth-sdxl-1.0", "diffusion_pytorch_model.safetensors",
     "controlnet/controlnet-depth-sdxl.safetensors", 2.33),
    ("video", "Meina/MeinaMix", "Meina V9 - baked VAE.safetensors",
     "checkpoints/meinamix_v9.safetensors", 3.74),
    ("video", "guoyww/animatediff", "mm_sd_v15_v2.ckpt",
     "animatediff/mm_sd_v15_v2.ckpt", 1.69),
]


def make_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def download(opener, repo: str, path: str, dest: Path, expect_gb: float) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MIRROR}/{repo}/resolve/main/{path.replace(' ', '%20')}?download=true"
    expect = int(expect_gb * 1024 ** 3)
    have = dest.stat().st_size if dest.exists() else 0
    if have and abs(have - expect) < expect * 0.03:
        print(f"  skip (complete): {dest.name} {have/1024**3:.2f}GB", flush=True)
        return True

    req = urllib.request.Request(url, headers={"User-Agent": "dsh-downloader/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    t0 = time.time()
    try:
        with opener.open(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0) + have
            mode = "ab" if have and r.status == 206 else "wb"
            if mode == "wb":
                have = 0
            done = have
            last = 0.0
            with open(dest, mode) as f:
                while True:
                    chunk = r.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last > 5:
                        last = now
                        speed = (done - have) / max(1e-6, now - t0) / 1024 ** 2
                        pct = (done / total * 100) if total else 0
                        print(f"  {dest.name}: {done/1024**3:.2f}/{total/1024**3:.2f}GB "
                              f"({pct:.1f}%) {speed:.1f}MB/s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {repo}/{path}: {e}", flush=True)
        return False

    got = dest.stat().st_size
    ok = got > 0 and (expect == 0 or got > expect * 0.9)
    print(f"  done {dest.name}: {got/1024**3:.2f}GB {'OK' if ok else 'SIZE-MISMATCH'}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["core", "extra", "video", "all"], default="all")
    args = ap.parse_args()

    opener = make_opener()
    sel = [t for t in TASKS if args.only == "all" or t[0] == args.only]
    print(f"downloading {len(sel)} files, total ~{sum(t[4] for t in sel):.1f}GB", flush=True)

    fail = []
    for group, repo, path, rel, gb in sel:
        dest = MODELS / rel
        print(f"[{group}] {repo} :: {path} -> {rel}", flush=True)
        for attempt in range(1, 4):
            if download(opener, repo, path, dest, gb):
                break
            print(f"  retry {attempt}/3 ...", flush=True)
            time.sleep(3)
        else:
            fail.append(rel)

    print("\n=== summary ===", flush=True)
    for group, repo, path, rel, gb in sel:
        d = MODELS / rel
        ok = d.exists() and d.stat().st_size > gb * 0.9 * 1024 ** 3
        print(f"  {'OK  ' if ok else 'MISS'} {rel} ({d.stat().st_size/1024**3:.2f}GB)" if d.exists() else f"  MISS {rel}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
