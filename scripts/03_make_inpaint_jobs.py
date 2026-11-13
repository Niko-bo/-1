"""Phase 4: 生成「表情/局部重绘」任务单，交给 ComfyUI 批量跑。

思路：不动身体，只重绘 1488x2272 画布上那个固定面部矩形（外扩 14px 做融合羽化）。
这样任何新表情都能与现有立绘、现有差分层共存，不会错位。

产出：
  out/inpaint/jobs.jsonl     每个任务：底图、遮罩、正/负提示词、seed、输出名
  out/inpaint/masks/*.png    遮罩（白=重绘区，黑=保留）
  out/inpaint/prompt_card.md 提示词表，方便手动跑或改

用法：
  python scripts/03_make_inpaint_jobs.py --bases "out/dataset/char/*.png" --group 常装1
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
CANVAS = (1488, 2272)
FACE_RECT = (782, 403, 928, 500)   # 实测：所有差分的并集
FEATHER = 14

# 表情预设 = 眼部 + 嘴部 + 情绪辅助词 + 权重建议
PRESETS = {
    "smile":        dict(eyes="gentle smile, eyes slightly narrowed", mouth="closed mouth smile, soft smile",
                         extra="warm expression, subtle blush", strength=0.55),
    "smile_open":   dict(eyes="happy eyes, curved lower eyelids", mouth="open mouth, teeth visible, joyful smile",
                         extra="bright expression", strength=0.6),
    "serious":      dict(eyes="sharp gaze, calm eyes", mouth="closed mouth, neutral lips",
                         extra="composed, stern", strength=0.5),
    "surprised":    dict(eyes="wide eyes, raised eyebrows", mouth="small open mouth, o shape",
                         extra="surprised, sweat drop", strength=0.6),
    "angry":        dict(eyes="furrowed brows, glaring", mouth="pouting, gritted teeth",
                         extra="annoyed, blush on cheeks", strength=0.6),
    "sad":          dict(eyes="teary eyes, downturned brows", mouth="trembling lips",
                         extra="sad, holding back tears", strength=0.6),
    "cry":          dict(eyes="closed eyes, tears streaming", mouth="open mouth, sobbing",
                         extra="tears, wet cheeks", strength=0.65),
    "shy":          dict(eyes="averted gaze, half-closed eyes", mouth="small closed mouth",
                         extra="heavy blush, embarrassed, looking away", strength=0.65),
    "half_closed":  dict(eyes="half-closed eyes, lidded eyes", mouth="slight smile",
                         extra="sultry, relaxed expression", strength=0.6),
    "closed_eyes":  dict(eyes="closed eyes, relaxed eyelids", mouth="gentle smile",
                         extra="peaceful", strength=0.55),
    "blank":        dict(eyes="empty eyes, dull eyes", mouth="slightly open mouth",
                         extra="expressionless, hollow", strength=0.65),
    "pant":         dict(eyes="half-closed eyes, teary eyes", mouth="open mouth, panting",
                         extra="heavy blush, sweat on face, messy hair strands", strength=0.7),
}

NEG = ("lowres, bad anatomy, bad hands, extra fingers, deformed face, asymmetric eyes, "
       "blurry, jpeg artifacts, watermark, text, signature, extra limbs, "
       "different character, wrong eye color, wrong hair color, "
       "seam at face edge, visible mask border")


def make_mask(path: Path, rect=FACE_RECT, feather=FEATHER, canvas=CANVAS):
    m = Image.new("L", canvas, 0)
    d = ImageDraw.Draw(m)
    x0, y0, x1, y1 = rect
    d.rectangle([x0 - feather, y0 - feather, x1 + feather, y1 + feather], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(3))
    # 单通道遮罩
    out = Image.merge("RGB", (m, m, m))
    out.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default=str(ROOT / "out" / "dataset" / "char" / "*.png"))
    ap.add_argument("--presets", nargs="*", default=list(PRESETS))
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--only", nargs="*", default=None, help="只处理文件名包含这些子串的底图")
    args = ap.parse_args()

    out = ROOT / "out" / "inpaint"
    mask_dir = out / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    bases = sorted(Path(p) for p in glob.glob(args.bases))
    if args.only:
        bases = [b for b in bases if any(k in b.name for k in args.only)]
    if not bases:
        print("no base images found")
        return 2

    mask_path = mask_dir / "face_mask_1488x2272.png"
    make_mask(mask_path)

    jobs, rows = [], ["| 底图 | 表情 | strength | 正提示词 |", "|---|---|---|---|"]
    i = 0
    for b in bases:
        for name in args.presets:
            p = PRESETS[name]
            prompt = (f"{p['eyes']}, {p['mouth']}, {p['extra']}, "
                      "silver hair bangs, blue eyes, detailed face, anime style, "
                      "same character, consistent lighting")
            jobs.append({
                "base": str(b), "mask": str(mask_path),
                "positive": prompt, "negative": NEG,
                "seed": args.seed + i, "denoise": p["strength"],
                "out": str(out / f"{b.stem}__{name}.png"),
                "face_rect": list(FACE_RECT),
            })
            rows.append(f"| {b.stem} | {name} | {p['strength']} | {prompt} |")
            i += 1

    (out / "jobs.jsonl").write_text(
        "\n".join(json.dumps(j, ensure_ascii=False) for j in jobs), encoding="utf-8")
    (out / "prompt_card.md").write_text("\n".join(rows), encoding="utf-8")

    print(f"bases   : {len(bases)}")
    print(f"presets : {len(args.presets)}")
    print(f"jobs    : {len(jobs)}  -> {out / 'jobs.jsonl'}")
    print(f"mask    : {mask_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
