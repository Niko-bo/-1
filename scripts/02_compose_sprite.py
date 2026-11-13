"""Phase 3/4: 把 AI 产出的「底图 + 表情层/服装层」合成为游戏可直接用的立绘。

为什么需要它：你的资产规范是 1488x2272 固定画布 + 面部差分层固定坐标。
AI 生成的图尺寸/对位都不可靠，必须先对齐再合成，否则进游戏会错位。

用法：
  # 单张合成
  python scripts/02_compose_sprite.py --base out/x/base.png --face out/x/face_01.png --out out/x/final.png

  # 批量：--faces 可给多个，输出多张 + 一张审图sheet
  python scripts/02_compose_sprite.py --base base.png --faces f1.png f2.png --outdir out/set1 --sheet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS = (1488, 2272)
# 从参考素材统计出的面部差分区（见 out/report.md）
FACE_RECT = (782, 403, 928, 500)


def load_rgba(p: Path) -> Image.Image:
    return Image.open(p).convert("RGBA")


def normalize_canvas(img: Image.Image, canvas=CANVAS) -> Image.Image:
    """缩放到统一画布；保持人物脚底对齐（底部对齐），避免上下漂。"""
    if img.size == canvas:
        return img
    k = min(canvas[0] / img.width, canvas[1] / img.height)
    nw, nh = int(round(img.width * k)), int(round(img.height * k))
    resized = img.resize((nw, nh), Image.LANCZOS)
    out = Image.new("RGBA", canvas, (0, 0, 0, 0))
    out.paste(resized, ((canvas[0] - nw) // 2, canvas[1] - nh))
    return out


def check_patch(patch: Image.Image) -> dict:
    """表情层是否落在规范面部矩形内（这才是"会不会错位"的真实判据）。"""
    b = patch.getchannel("A").getbbox()
    if b is None:
        return {"bbox": None, "inside": False, "offset": None}
    cx, cy = (FACE_RECT[0] + FACE_RECT[2]) / 2, (FACE_RECT[1] + FACE_RECT[3]) / 2
    px, py = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    inside = (b[0] >= FACE_RECT[0] - 24 and b[1] >= FACE_RECT[1] - 24
              and b[2] <= FACE_RECT[2] + 24 and b[3] <= FACE_RECT[3] + 24)
    return {"bbox": list(b), "inside": inside, "offset": [round(px - cx), round(py - cy)]}


def make_sheet(images: list[tuple[str, Image.Image]], out: Path, cell=240):
    cols = min(6, max(1, len(images)))
    rows = (len(images) + cols - 1) // cols
    lab = 16
    sheet = Image.new("RGB", (cols * cell, rows * (cell + lab)), (238, 238, 240))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    for i, (name, im) in enumerate(images):
        r, c = divmod(i, cols)
        k = min(cell / im.width, (cell - 4) / im.height)
        th = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
        bg = Image.new("RGB", (cell, cell), (255, 255, 255))
        bg.paste(th, ((cell - th.width) // 2, (cell - th.height) // 2), th)
        sheet.paste(bg, (c * cell, r * (cell + lab)))
        d.text((c * cell + 3, r * (cell + lab) + cell + 2), name[:32], fill=(20, 20, 20), font=font)
    sheet.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--face", nargs="*", default=[])
    ap.add_argument("--faces", nargs="*", default=[], help="alias of --face")
    ap.add_argument("--out", default=None, help="单张输出路径")
    ap.add_argument("--outdir", default=None, help="批量输出目录")
    ap.add_argument("--sheet", action="store_true", help="生成审图总览")
    ap.add_argument("--zoom", type=int, default=0, help="额外输出面部放大校验图")
    ap.add_argument("--json", default=None, help="输出校验报告 JSON")
    args = ap.parse_args()

    base_path = Path(args.base)
    base = normalize_canvas(load_rgba(base_path))
    patches = [Path(p) for p in (args.face + args.faces)]
    outdir = Path(args.outdir) if args.outdir else (Path(args.out).parent if args.out else base_path.parent)
    outdir.mkdir(parents=True, exist_ok=True)

    if not patches:
        out = Path(args.out) if args.out else outdir / f"{base_path.stem}_norm.png"
        base.save(out)
        print(f"normalized -> {out}")
        return 0

    report, sheet_items = [], []
    for p in patches:
        patch = load_rgba(p)
        issues = []
        if patch.size != base.size:
            issues.append(f"size {patch.size} != canvas {base.size} -> 已按底部对齐归一化")
            patch = normalize_canvas(patch)
        chk = check_patch(patch)
        issues = []
        if not chk["inside"]:
            issues.append(f"表情层超出规范面部区 bbox={chk['bbox']}，合成可能错位")

        merged = base.copy()
        merged.alpha_composite(patch)

        name = f"{base_path.stem}__{p.stem}.png"
        outp = Path(args.out) if (args.out and len(patches) == 1) else outdir / name
        merged.save(outp)
        sheet_items.append((p.stem, merged))

        if args.zoom:
            z = merged.crop(FACE_RECT)
            z = z.resize((z.width * args.zoom, z.height * args.zoom), Image.LANCZOS)
            z.save(outp.with_name(outp.stem + "_facezoom.png"))

        report.append({"base": str(base_path), "patch": str(p), "out": str(outp),
                       "canvas": list(merged.size), "patch_bbox": chk["bbox"],
                       "inside_face_rect": chk["inside"], "offset": chk["offset"],
                       "issues": issues})
        print(f"composed -> {outp}" + (f"  [!] {'; '.join(issues)}" if issues else ""))

    if args.sheet and sheet_items:
        s = outdir / "sheet.png"
        make_sheet(sheet_items, s)
        print(f"sheet -> {s}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
