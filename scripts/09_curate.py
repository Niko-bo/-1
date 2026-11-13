"""Phase 1.5-d: 把人工挑中的候选图变成正式训练集（带打标与构成统计）。

用法：
  # 1) 先看审图表 out/qa/gen15/<base>.png，把要的图名写进 picks.txt（一行一个，可省 .png）
  python scripts/09_curate.py --picks picks.txt
  # 2) 也可以直接给名字
  python scripts/09_curate.py --names a__b__01_standing_front__d90__s123 a__c__04_sitting_chair__d90__s456

产出：
  out/dataset/final/           挑中的图 + 同名 .txt 打标
  out/dataset/final_report.md  构成统计（单人与场景比例、姿势覆盖）
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

POSE_TAGS = {
    "01_standing_front": "standing, facing viewer, arms at sides",
    "02_three_quarter_hip": "standing, hand on hip, cowboy shot",
    "03_hands_behind": "standing, hands behind back",
    "04_sitting_chair": "sitting, on chair, knees together",
    "05_sitting_floor": "sitting, hugging own knees, on floor",
    "06_kneeling": "kneeling",
    "07_leaning_wall": "leaning back, against wall",
    "08_back_view": "from behind, back to viewer",
    "09_side_profile": "from side, profile",
    "10_bust_portrait": "upper body, portrait, looking at viewer",
    "11_looking_back": "looking back, over shoulder",
    "12_arms_up": "standing, arms up, stretching",
}

# caption 里复用底图标签时需要剔掉的词（姿势/场景/数量）
DROP = {"standing", "full body", "looking at viewer", "solo", "solo focus",
        "simple background", "white background", "indoors", "wooden wall",
        "cinematic lighting", "multiple boys", "faceless male", "adult",
        "upper body", "cowboy shot", "against wall", "from behind", "from side",
        "sitting", "kneeling", "grabbed from behind", "crowded", "depth of field"}


def find_base(base_dir: Path, base_stem: str) -> Path | None:
    """char/ 里的文件叫 <group>__<原名>.png，按前缀找回去。"""
    exact = base_dir / f"{base_stem}.png"
    if exact.exists():
        return exact
    for p in base_dir.glob("*.png"):
        if p.stem == base_stem or p.stem.startswith(f"{base_stem}__"):
            return p
    return None


def char_tags_from(path: Path) -> list[str]:
    txt = path.with_suffix(".txt")
    if not txt.exists():
        return ["nmk_char", "1girl", "silver hair", "very long hair", "hair over one eye",
                "blue eyes", "mature female", "large breasts"]
    tags = [t.strip() for t in txt.read_text(encoding="utf-8").split(",") if t.strip()]
    keep = [t for t in tags if t.lower() not in DROP]
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", default=None, help="文件名清单，一行一个")
    ap.add_argument("--names", nargs="*", default=[])
    ap.add_argument("--src", default=str(ROOT / "out" / "gen15"))
    ap.add_argument("--base", default=str(ROOT / "out" / "dataset" / "char"))
    ap.add_argument("--out", default=str(ROOT / "out" / "dataset" / "final"))
    args = ap.parse_args()

    wanted = list(args.names)
    if args.picks:
        for line in Path(args.picks).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                wanted.append(line)
    if not wanted:
        print("no picks given")
        return 2

    src = Path(args.src)
    base_dir = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.png")) + list(out.glob("*.txt")):
        old.unlink(missing_ok=True)

    missing, done, poses = [], 0, Counter()
    for name in wanted:
        stem = Path(name).stem
        f = src / f"{stem}.png"
        if not f.exists():
            missing.append(name)
            continue
        parts = stem.split("__")
        base_stem = parts[0] if parts else stem
        pose = parts[2] if len(parts) > 2 else ""
        shutil.copy2(f, out / f"{stem}.png")

        tags = char_tags_from(find_base(base_dir, base_stem) or (base_dir / f"{base_stem}.png"))
        tags += [t.strip() for t in POSE_TAGS.get(pose, "").split(",") if t.strip()]
        tags += ["simple background", "white background", "full body" if pose != "10_bust_portrait" else "upper body"]
        caption = ", ".join(dict.fromkeys(tags))
        (out / f"{stem}.txt").write_text(caption, encoding="utf-8")
        poses[pose or "unknown"] += 1
        done += 1

    total_sprites = sum(1 for p in base_dir.glob("*.png"))
    lines = ["# 阶段 1.5 交付集", "",
             f"- 入选：{done} 张（清单外的 {len(missing)} 个名字没找到）",
             f"- 单人立绘基数：{total_sprites} 张（来自 out/dataset/char）",
             "", "## 姿势覆盖", ""]
    for k, v in sorted(poses.items()):
        lines.append(f"- `{k}`：{v} 张")
    lines += ["", "## 训练集构成建议", "",
              f"- 现有：{done} 张生成图 + {total_sprites} 张原始立绘 = {done + total_sprites} 张",
              "- 建议原始单人立绘占比 ≥50%，否则 LoRA 会偏向生成图的脸",
              ""]
    (out.parent / "final_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"selected : {done}")
    print(f"missing  : {len(missing)}" + (f" -> {missing[:5]}" if missing else ""))
    print("poses    : " + "  ".join(f"{k}:{v}" for k, v in sorted(poses.items())))
    print(f"out      : {out}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
