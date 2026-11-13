"""Phase 1: 按新数据集结构整理训练素材（151 个文件 / 4 个子目录）。

分类：
  人物初版设计图  -> sprite / sprite_multi（双人）/ event_cg
  表情            -> face_patch（面部小层）/ sprite（服装底图）
  姿势训练        -> pose_cg / face_patch（go_f* 这类场景表情层）
  骚扰分镜        -> storyboard（参考用）/ layer（手部、遮罩等局部件）

产出：
  out/dataset/char/        纯单人立绘（最干净的身份集）
  out/dataset/char_plus/   单人立绘 + 双人立绘 + 场景图 + 事件 CG（补姿势）
  out/dataset/faces/       表情差分裁切
  out/dataset/pairs.json   差分画布与坐标
  out/dataset/_review.csv  需人工复核的打标清单
  out/qa/*.png             各类目审图
  out/report.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

# 立绘边缘有 alpha 很低的噪点，getbbox 会把包围盒撑到画布边缘，必须按阈值取实心区域
ALPHA_THR = 96
LAYER_HINT = re.compile(r"(_hand|_mask|mask_|_work|_kage)", re.I)
MULTI_RATIO = 0.45   # 实心包围盒宽 / 画布宽 超过此值 -> 画面里还有第二个人


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def solid_bbox(img: Image.Image, thr: int = ALPHA_THR):
    a = np.asarray(img.getchannel("A"))
    ys, xs = np.nonzero(a > thr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def group_key(stem: str) -> str:
    s = re.sub(r"\s*-\s*コピー.*$", "", stem)
    s = re.sub(r"_(default|\d+)$", "", s)
    return s


def base_tags(trigger: str) -> list[str]:
    return [trigger, "1girl", "silver hair", "very long hair", "hair over one eye",
            "blue eyes", "mature female", "large breasts", "tall", "detailed eyes"]


def caption_for(kind: str, stem: str, trigger: str) -> str:
    t = base_tags(trigger)
    if kind == "sprite":
        t += ["solo", "full body", "standing", "looking at viewer",
              "simple background", "white background"]
        if stem.startswith(("常装", "lob")):
            t += ["white dress shirt", "black pencil skirt", "long cape", "dark blue cape",
                  "gold trim", "black pantyhose", "brown belt"]
        elif stem.startswith("内衣"):
            t += ["black lace lingerie", "open white shirt", "garter belt", "black thighhighs"]
        elif stem.startswith("nugi"):
            t += ["undressed", "nude", "bare shoulders"]
        elif stem.startswith("huku"):
            # hukusimi 是白衬衫 + 黑裙 + 黑丝（与常装同套），不是制服
            t += ["white dress shirt", "black pencil skirt", "black pantyhose"]
        else:
            t += ["white dress shirt", "black pencil skirt", "black pantyhose"]
    elif kind == "sprite_multi":
        t += ["solo focus", "1boy", "faceless male", "multiple boys",
              "white dress shirt", "black pencil skirt", "black pantyhose",
              "simple background", "white background", "adult"]
    elif kind == "event_cg":
        t += ["solo", "full body", "standing", "indoors", "cinematic lighting",
              "depth of field", "looking at viewer"]
    elif kind == "pose_cg":
        t += ["solo focus", "indoors", "wooden wall", "cinematic lighting",
              "multiple boys", "faceless male", "adult"]
        if stem.startswith("aibu"):
            t += ["standing", "against wall", "open white shirt", "black lingerie",
                  "black thighhighs"]
        elif stem.startswith("kunni"):
            t += ["sitting", "kneeling", "white dress shirt", "black skirt"]
        elif stem.startswith("go_momi"):
            t += ["upper body", "white dress shirt", "black skirt", "grabbed from behind"]
        elif stem.startswith("go_sekuhara"):
            t += ["standing", "white dress shirt", "black skirt", "crowded"]
    return ", ".join(dict.fromkeys(t))


def fit_bucket(img: Image.Image, box, bucket, pad_ratio: float = 0.03):
    w, h = img.size
    x0, y0, x1, y1 = box
    pad = int(max(x1 - x0, y1 - y0) * pad_ratio)
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
    crop = img.crop((x0, y0, x1, y1))
    tw, th = bucket
    ratio = tw / th
    cw, ch = crop.size
    if cw / ch > ratio:
        nw, nh = cw, int(round(cw / ratio))
    else:
        nw, nh = int(round(ch * ratio)), ch
    canvas = Image.new("RGBA", (nw, nh), (0, 0, 0, 0))
    canvas.paste(crop, ((nw - cw) // 2, (nh - ch) // 2))
    canvas = canvas.resize((tw, th), Image.LANCZOS)
    bg = Image.new("RGB", (tw, th), (255, 255, 255))
    bg.paste(canvas, (0, 0), canvas)
    return bg


def sheet(items, out: Path, cols: int = 8, cell: int = 150):
    if not items:
        return None
    rows = (len(items) + cols - 1) // cols
    lab = 15
    s = Image.new("RGB", (cols * cell, rows * (cell + lab)), (235, 235, 238))
    d = ImageDraw.Draw(s)
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
    for i, (label, im) in enumerate(items):
        r, c = divmod(i, cols)
        k = min(cell / im.width, (cell - 4) / im.height)
        th = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
        bg = Image.new("RGB", (cell, cell), (255, 255, 255))
        if th.mode == "RGBA":
            bg.paste(th, ((cell - th.width) // 2, (cell - th.height) // 2), th)
        else:
            bg.paste(th, ((cell - th.width) // 2, (cell - th.height) // 2))
        s.paste(bg, (c * cell, r * (cell + lab)))
        d.text((c * cell + 2, r * (cell + lab) + cell + 1), label[:26], fill=(20, 20, 20), font=font)
    s.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "paths.json"))
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    trigger = cfg["trigger_token"]
    bucket = tuple(cfg["lora_bucket"])
    face_pad = int(cfg["face_pad"])

    dirs = {k: Path(cfg[k]) for k in ("sprite_dir", "face_dir", "pose_dir", "cg_dir") if cfg.get(k)}
    out = Path(cfg["out_dir"])
    char_dir = out / "dataset" / "char"
    plus_dir = out / "dataset" / "char_plus"
    face_out = out / "dataset" / "faces"
    qa = out / "qa"
    for d in (char_dir, plus_dir, face_out, qa):
        d.mkdir(parents=True, exist_ok=True)
        for old in list(d.glob("*.png")) + list(d.glob("*.txt")):
            old.unlink(missing_ok=True)

    records = []
    seen: dict[str, str] = {}
    for tag, d in dirs.items():
        if not d.exists():
            print(f"WARN missing dir {d}")
            continue
        for p in sorted(d.glob("*.png")):
            try:
                img = Image.open(p).convert("RGBA")
            except Exception as e:  # noqa: BLE001
                records.append({"file": p, "tag": tag, "kind": "error", "error": str(e)})
                continue
            box = solid_bbox(img)
            w, h = img.size
            digest = sha256(p)
            if box is None:
                kind = "empty"
            elif LAYER_HINT.search(p.stem):
                kind = "layer"
            else:
                bw, bh = box[2] - box[0], box[3] - box[1]
                multi = bw > w * MULTI_RATIO
                if bw <= 400 and bh <= 400:
                    kind = "face_patch"
                elif tag == "sprite_dir":
                    kind = "event_cg" if w / h >= 1.3 else ("sprite_multi" if multi else "sprite")
                elif tag == "face_dir":
                    kind = "sprite_multi" if multi else "sprite"
                elif tag == "pose_dir":
                    kind = "pose_cg"
                else:
                    kind = "storyboard"
            records.append({"file": p, "tag": tag, "kind": kind, "w": w, "h": h,
                            "box": box, "size": p.stat().st_size, "sha": digest,
                            "dup": seen.get(digest), "group": group_key(p.stem)})
            seen.setdefault(digest, p.name)
            img.close()

    kinds: dict[str, int] = {}
    for r in records:
        kinds[r.get("kind", "error")] = kinds.get(r.get("kind", "error"), 0) + 1

    review = ["file,kind,group,goes_to,caption,flag"]
    written = {"char": 0, "char_plus": 0}

    def emit(r, target_dir: Path, setname: str):
        p = Path(r["file"])
        img = Image.open(p).convert("RGBA")
        try:
            fitted = fit_bucket(img, r["box"], bucket)
        finally:
            img.close()
        name = f"{r['group']}__{p.stem}".replace(" ", "_").replace("-", "")
        fitted.save(target_dir / f"{name}.png")
        cap = caption_for(r["kind"], p.stem, trigger)
        (target_dir / f"{name}.txt").write_text(cap, encoding="utf-8")
        written[setname] += 1
        flag = ""
        if p.stem.startswith(("huku", "nugi0[1-3]")):
            flag = "REVIEW_OUTFIT"
        if r["kind"] in ("pose_cg", "sprite_multi"):
            flag = (flag + "|NSFW_SCENE").strip("|")
        review.append(f'{p.name},{r["kind"]},{r["group"]},{setname},"{cap}",{flag}')

    for r in records:
        k = r.get("kind")
        if r.get("dup"):
            continue
        if k == "sprite":
            emit(r, char_dir, "char")
            emit(r, plus_dir, "char_plus")
        elif k in ("pose_cg", "event_cg", "sprite_multi"):
            emit(r, plus_dir, "char_plus")

    patches = [r for r in records if r.get("kind") == "face_patch" and not r.get("dup")]
    pairs, items = [], []
    for r in patches:
        p = Path(r["file"])
        img = Image.open(p).convert("RGBA")
        b = r["box"]
        crop = img.crop((max(0, b[0] - face_pad), max(0, b[1] - face_pad),
                         min(img.width, b[2] + face_pad), min(img.height, b[3] + face_pad)))
        name = f"{p.stem.replace(' ', '_').replace('-', '')}.png"
        crop.save(face_out / name)
        items.append((p.stem, crop))
        pairs.append({"source": str(p), "canvas": [img.width, img.height],
                      "alpha_bbox": b, "group": r["group"], "crop": str(face_out / name)})
        img.close()

    canvases: dict[str, dict] = {}
    for pr in pairs:
        key = f"{pr['canvas'][0]}x{pr['canvas'][1]}"
        c = canvases.setdefault(key, {"count": 0, "groups": set(),
                                      "bbox_union": [10 ** 6, 10 ** 6, -1, -1]})
        c["count"] += 1
        c["groups"].add(pr["group"])
        b = pr["alpha_bbox"]
        c["bbox_union"] = [min(c["bbox_union"][0], b[0]), min(c["bbox_union"][1], b[1]),
                           max(c["bbox_union"][2], b[2]), max(c["bbox_union"][3], b[3])]
    for k in canvases:
        canvases[k]["groups"] = sorted(canvases[k]["groups"])

    (out / "dataset" / "pairs.json").write_text(
        json.dumps({"canvases": canvases, "pairs": pairs}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out / "dataset" / "_review.csv").write_text("\n".join(review), encoding="utf-8")

    sheet([(p.stem, Image.open(p).convert("RGBA")) for p in sorted(char_dir.glob("*.png"))],
          qa / "sprites.png", cols=8, cell=150)
    sheet([(p.stem, Image.open(p).convert("RGBA")) for p in sorted(plus_dir.glob("*.png"))],
          qa / "char_plus.png", cols=9, cell=130)
    sheet(sorted(items, key=lambda z: z[0]), qa / "faces.png", cols=10, cell=110)
    sheet([(r["file"].stem, Image.open(r["file"]).convert("RGBA"))
           for r in records if r.get("kind") == "storyboard"], qa / "storyboard.png",
          cols=10, cell=120)

    lines = ["# 素材分类报告（新数据集）", "",
             f"- 扫描：{len(records)} 个文件", ""]
    for k, v in sorted(kinds.items()):
        lines.append(f"- `{k}`：{v}")
    lines += ["", "## 训练集", "",
              f"- `out/dataset/char/`：{written['char']} 张（纯单人立绘，身份训练主力）",
              f"- `out/dataset/char_plus/`：{written['char_plus']} 张（+ 双人/场景/事件CG，补姿势）",
              f"- `out/dataset/faces/`：{len(pairs)} 张唯一表情差分", "",
              "## 差分画布规格", ""]
    for k, v in canvases.items():
        u = v["bbox_union"]
        lines.append(f"- `{k}`：{v['count']} 张，组：{', '.join(v['groups'])}")
        lines.append(f"  - 面部区并集：x={u[0]}..{u[2]}, y={u[1]}..{u[3]}")
    lines += ["", "## 重复文件（哈希相同）", ""]
    bysha: dict[str, list[str]] = {}
    for r in records:
        if r.get("sha"):
            bysha.setdefault(r["sha"], []).append(r["file"].name)
    for names in bysha.values():
        if len(names) > 1:
            lines.append("- " + " == ".join(names))
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("=== phase1 report ===")
    print(f"scanned     : {len(records)}")
    for k, v in sorted(kinds.items()):
        print(f"  {k:<12}: {v}")
    print(f"char/       : {written['char']}")
    print(f"char_plus/  : {written['char_plus']}")
    print(f"faces/      : {len(pairs)} unique")
    for k, v in canvases.items():
        print(f"  canvas {k}: {v['count']} patches")
    print(f"report      : {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
