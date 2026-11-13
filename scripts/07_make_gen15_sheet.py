"""Phase 1.5-c: 把生成的候选图做成审图表，并自动标记明显失败的图。

自动标记（只挡掉明显废图，不替代人工挑）：
  blank   颜色方差过低（一片白/一片灰）
  dark    平均亮度 < 30
  blurry  边缘能量过低
  tiny    文件 < 20KB

产出：
  out/qa/gen15/<base>.png   每个底图一张表，第一格放原立绘做对照，其余按姿势排
  out/qa/gen15/_all.png     总览
  out/qa/gen15_flags.csv    每张图的指标与标记
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

ROOT = Path(__file__).resolve().parents[1]


def stats(path: Path) -> dict:
    img = Image.open(path).convert("RGB")
    g = img.convert("L")
    st = ImageStat.Stat(g)
    edges = g.filter(ImageFilter.FIND_EDGES)
    # 白底生成图：非白像素的包围盒贴边 -> 人物被裁切
    mask = g.point(lambda v: 255 if v < 245 else 0)
    box = mask.getbbox()
    touch = 0
    if box:
        w, h = img.size
        if box[0] <= 2:
            touch += 1
        if box[1] <= 2:
            touch += 1
        if box[2] >= w - 3:
            touch += 1
        if box[3] >= h - 3:
            touch += 1
    return {
        "w": img.width, "h": img.height,
        "mean": round(st.mean[0], 1),
        "std": round(st.stddev[0], 1),
        "edge": round(ImageStat.Stat(edges).stddev[0], 1),
        "kb": round(path.stat().st_size / 1024, 1),
        "border_touch": touch,
    }


def flags(s: dict) -> list[str]:
    f = []
    if s["std"] < 8:
        f.append("blank")
    if s["mean"] < 30:
        f.append("dark")
    if s["edge"] < 10:
        f.append("blurry")
    if s["kb"] < 20:
        f.append("tiny")
    if s["border_touch"] >= 2:
        f.append("cropped")
    return f


def label_font(size: int = 10):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def grid(entries, out: Path, cols=4, cell=260):
    if not entries:
        return None
    rows = (len(entries) + cols - 1) // cols
    lab = 16
    sheet = Image.new("RGB", (cols * cell, rows * (cell + lab)), (232, 232, 236))
    d = ImageDraw.Draw(sheet)
    font = label_font()
    for i, (label, path, bad) in enumerate(entries):
        r, c = divmod(i, cols)
        im = Image.open(path).convert("RGB")
        k = min((cell - 6) / im.width, (cell - 6) / im.height)
        th = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
        x = c * cell + (cell - th.width) // 2
        y = r * (cell + lab) + (cell - th.height) // 2
        sheet.paste(th, (x, y))
        if bad:
            d.rectangle([c * cell, r * (cell + lab), (c + 1) * cell - 1, (r + 1) * (cell + lab) - 1],
                        outline=(200, 30, 30), width=2)
        d.text((c * cell + 3, r * (cell + lab) + cell + 2),
               (label + ("  !" + ",".join(bad) if bad else ""))[:40],
               fill=(160, 20, 20) if bad else (20, 20, 20), font=font)
    sheet.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "out" / "gen15"))
    ap.add_argument("--refs", default=str(ROOT / "out" / "dataset" / "char"))
    ap.add_argument("--out", default=str(ROOT / "out" / "qa" / "gen15"))
    args = ap.parse_args()

    gen_dir = Path(args.dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(gen_dir.glob("*.png"))
    if not files:
        print(f"no images in {gen_dir}")
        return 2

    rows, by_base = [], {}
    for p in files:
        parts = p.stem.split("__")
        base = parts[0] if parts else p.stem
        pose = parts[1] if len(parts) > 1 else ""
        s = stats(p)
        f = flags(s)
        by_base.setdefault(base, []).append((pose, p, f))
        rows.append({"file": p.name, "base": base, "pose": pose, **s, "flags": "|".join(f)})

    refs = {p.stem.split("__")[0]: p for p in Path(args.refs).glob("*.png")}
    made = []
    for base, items in sorted(by_base.items()):
        items.sort(key=lambda z: z[0])
        entry = []
        if base in refs:
            entry.append(("ORIGINAL " + base, refs[base], []))
        entry += items
        made.append(grid(entry, out / f"{base}.png", cols=4, cell=260))

    all_entries = []
    for base, items in sorted(by_base.items()):
        for pose, p, f in items:
            all_entries.append((f"{base}::{pose}", p, f))
    grid(all_entries, out / "_all.png", cols=5, cell=210)

    csv_path = out / "gen15_flags.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    bad = sum(1 for r in rows if r["flags"])
    by_pose = {}
    for r in rows:
        by_pose.setdefault(r["pose"], [0, 0])
        by_pose[r["pose"]][0] += 1
        if r["flags"]:
            by_pose[r["pose"]][1] += 1
    print(f"images   : {len(rows)}")
    print(f"flagged  : {bad}")
    print("poses    : " + "  ".join(f"{k}:{v[0]}(-{v[1]})" for k, v in sorted(by_pose.items())))
    print(f"sheets   : {len([m for m in made if m])} -> {out}")
    print(f"csv      : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
