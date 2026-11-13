"""Phase 1.5-a: 生成 OpenPose(18点) 骨架参考图，供 ControlNet 锁定姿势。

为什么自己画：从成品插画里跑 DW-Pose 会把旁边的第二个角色也一并检出，
骨架会串在一起；程序化骨架坐标精确，且能覆盖数据集里缺的角度。

产出：out/poses/<pose>.png  (832x1216，黑底彩线，OpenPose 标准配色)
      out/poses/_sheet.png  总览
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H = 832, 1216
LINE = 8
DOT = 7

# OpenPose 18 点顺序
# 0 nose 1 neck 2 Rsho 3 Relb 4 Rwri 5 Lsho 6 Lelb 7 Lwri
# 8 Rhip 9 Rknee 10 Rankl 11 Lhip 12 Lknee 13 Lankl
# 14 Reye 15 Leye 16 Rear 17 Lear
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
         (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
         (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
          (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
          (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
          (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def P(**kw):
    """pose = dict of 18 keypoints, keys: nose neck rsho relb rwri lsho lelb lwri
    rhip rknee rankl lhip lknee lankl reye leye rear lear (values (x,y) in 0..1)"""
    order = ["nose", "neck", "rsho", "relb", "rwri", "lsho", "lelb", "lwri",
             "rhip", "rknee", "rankl", "lhip", "lknee", "lankl",
             "reye", "leye", "rear", "lear"]
    # 允许缺关键点（半身姿势不画髋以下），缺失的返回 None
    return [kw.get(k) for k in order]


def head(cx, cy, r=0.032):
    """给一个头部中心，自动补鼻/眼/耳"""
    return {
        "nose": (cx, cy + r * 0.25), "reye": (cx - r * 0.45, cy - r * 0.15),
        "leye": (cx + r * 0.45, cy - r * 0.15), "rear": (cx - r * 0.9, cy + r * 0.1),
        "lear": (cx + r * 0.9, cy + r * 0.1),
    }


def build():
    poses: dict[str, list] = {}

    # 1 正面站姿
    p = head(0.50, 0.12)
    p |= dict(neck=(0.50, 0.20), rsho=(0.43, 0.22), lsho=(0.57, 0.22),
              relb=(0.41, 0.34), rwri=(0.40, 0.45),
              lelb=(0.59, 0.34), lwri=(0.60, 0.45),
              rhip=(0.46, 0.50), rknee=(0.455, 0.66), rankl=(0.45, 0.82),
              lhip=(0.54, 0.50), lknee=(0.545, 0.66), lankl=(0.55, 0.82))
    poses["01_standing_front"] = P(**p)

    # 2 三/四侧身，单手叉腰
    p = head(0.52, 0.12)
    p |= dict(neck=(0.52, 0.20), rsho=(0.45, 0.22), lsho=(0.61, 0.22),
              relb=(0.42, 0.33), rwri=(0.47, 0.44),
              lelb=(0.66, 0.32), lwri=(0.58, 0.40),
              rhip=(0.47, 0.50), rknee=(0.46, 0.66), rankl=(0.45, 0.82),
              lhip=(0.56, 0.50), lknee=(0.57, 0.66), lankl=(0.58, 0.82))
    poses["02_three_quarter_hip"] = P(**p)

    # 3 双手背后
    p = head(0.50, 0.12)
    p |= dict(neck=(0.50, 0.20), rsho=(0.44, 0.22), lsho=(0.56, 0.22),
              relb=(0.40, 0.33), rwri=(0.46, 0.42),
              lelb=(0.60, 0.33), lwri=(0.54, 0.42),
              rhip=(0.46, 0.50), rknee=(0.455, 0.66), rankl=(0.45, 0.82),
              lhip=(0.54, 0.50), lknee=(0.545, 0.66), lankl=(0.55, 0.82))
    poses["03_hands_behind"] = P(**p)

    # 4 坐姿（椅子）
    p = head(0.50, 0.16)
    p |= dict(neck=(0.50, 0.24), rsho=(0.44, 0.26), lsho=(0.56, 0.26),
              relb=(0.41, 0.38), rwri=(0.46, 0.48),
              lelb=(0.59, 0.38), lwri=(0.54, 0.48),
              rhip=(0.46, 0.52), rknee=(0.43, 0.64), rankl=(0.43, 0.80),
              lhip=(0.54, 0.52), lknee=(0.57, 0.64), lankl=(0.57, 0.80))
    poses["04_sitting_chair"] = P(**p)

    # 5 抱膝坐地
    p = head(0.48, 0.30)
    p |= dict(neck=(0.48, 0.38), rsho=(0.42, 0.40), lsho=(0.54, 0.40),
              relb=(0.38, 0.52), rwri=(0.44, 0.60),
              lelb=(0.58, 0.52), lwri=(0.52, 0.60),
              rhip=(0.44, 0.66), rknee=(0.38, 0.68), rankl=(0.40, 0.82),
              lhip=(0.52, 0.66), lknee=(0.58, 0.68), lankl=(0.56, 0.82))
    poses["05_sitting_floor"] = P(**p)

    # 6 跪姿
    p = head(0.50, 0.18)
    p |= dict(neck=(0.50, 0.26), rsho=(0.44, 0.28), lsho=(0.56, 0.28),
              relb=(0.42, 0.40), rwri=(0.45, 0.50),
              lelb=(0.58, 0.40), lwri=(0.55, 0.50),
              rhip=(0.46, 0.54), rknee=(0.45, 0.68), rankl=(0.45, 0.84),
              lhip=(0.54, 0.54), lknee=(0.55, 0.68), lankl=(0.55, 0.84))
    poses["06_kneeling"] = P(**p)

    # 7 靠墙（身体后倾，一手扶墙）
    p = head(0.46, 0.14)
    p |= dict(neck=(0.47, 0.22), rsho=(0.41, 0.24), lsho=(0.53, 0.24),
              relb=(0.33, 0.30), rwri=(0.28, 0.24),
              lelb=(0.58, 0.34), lwri=(0.62, 0.44),
              rhip=(0.45, 0.52), rknee=(0.44, 0.67), rankl=(0.42, 0.82),
              lhip=(0.52, 0.52), lknee=(0.53, 0.67), lankl=(0.55, 0.82))
    poses["07_leaning_wall"] = P(**p)

    # 8 抱臂站姿（双手交叉在胸前）
    p = head(0.50, 0.13)
    p |= dict(neck=(0.50, 0.21), rsho=(0.43, 0.23), lsho=(0.57, 0.23),
              relb=(0.36, 0.34), rwri=(0.47, 0.36),
              lelb=(0.64, 0.34), lwri=(0.53, 0.37),
              rhip=(0.46, 0.50), rknee=(0.455, 0.66), rankl=(0.45, 0.82),
              lhip=(0.54, 0.50), lknee=(0.545, 0.66), lankl=(0.55, 0.82))
    poses["08_arms_crossed"] = P(**p)

    # 9 一手抚胸、一手垂放
    p = head(0.50, 0.13)
    p |= dict(neck=(0.50, 0.21), rsho=(0.43, 0.23), lsho=(0.57, 0.23),
              relb=(0.38, 0.32), rwri=(0.47, 0.38),
              lelb=(0.62, 0.34), lwri=(0.62, 0.46),
              rhip=(0.46, 0.50), rknee=(0.455, 0.66), rankl=(0.45, 0.82),
              lhip=(0.54, 0.50), lknee=(0.545, 0.66), lankl=(0.55, 0.82))
    poses["09_hand_on_chest"] = P(**p)

    # 10 腰部以上（单手叉腰）
    p = head(0.50, 0.24, r=0.065)
    p |= dict(neck=(0.50, 0.34), rsho=(0.40, 0.39), lsho=(0.60, 0.39),
              relb=(0.33, 0.52), rwri=(0.44, 0.58),
              lelb=(0.68, 0.50), lwri=(0.58, 0.58),
              rhip=(0.45, 0.78), lhip=(0.55, 0.78))
    poses["10_waist_up"] = P(**p)

    # 11 回头（躯干转、头回看）
    p = head(0.42, 0.14)
    p |= dict(neck=(0.47, 0.21), rsho=(0.41, 0.23), lsho=(0.54, 0.24),
              relb=(0.39, 0.35), rwri=(0.38, 0.46),
              lelb=(0.60, 0.33), lwri=(0.64, 0.43),
              rhip=(0.46, 0.51), rknee=(0.45, 0.66), rankl=(0.44, 0.82),
              lhip=(0.55, 0.51), lknee=(0.56, 0.66), lankl=(0.57, 0.82))
    poses["11_looking_back"] = P(**p)

    # 12 伸展（双手上举）
    p = head(0.50, 0.11)
    p |= dict(neck=(0.50, 0.19), rsho=(0.44, 0.21), lsho=(0.56, 0.21),
              relb=(0.40, 0.12), rwri=(0.38, 0.03),
              lelb=(0.60, 0.12), lwri=(0.62, 0.03),
              rhip=(0.46, 0.49), rknee=(0.455, 0.65), rankl=(0.45, 0.82),
              lhip=(0.54, 0.49), lknee=(0.545, 0.65), lankl=(0.55, 0.82))
    poses["12_arms_up"] = P(**p)

    return poses


def render(pts, path: Path):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    xy = [None if p is None else (int(p[0] * W), int(p[1] * H)) for p in pts]
    for i, (a, b) in enumerate(LIMBS):
        if xy[a] is None or xy[b] is None:
            continue
        d.line([xy[a], xy[b]], fill=COLORS[i % len(COLORS)], width=LINE)
    for i, p in enumerate(xy):
        if p is None:
            continue
        c = COLORS[i % len(COLORS)]
        d.ellipse([p[0] - DOT, p[1] - DOT, p[0] + DOT, p[1] + DOT], fill=c)
    img.save(path)


def main():
    out = ROOT / "out" / "poses"
    out.mkdir(parents=True, exist_ok=True)
    poses = build()
    for name, pts in poses.items():
        render(pts, out / f"{name}.png")

    # 总览
    cols, cell = 6, 200
    rows = (len(poses) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 16)), (28, 28, 32))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    for i, (name, pts) in enumerate(poses.items()):
        r, c = divmod(i, cols)
        thumb = Image.open(out / f"{name}.png")
        k = min((cell - 6) / thumb.width, (cell - 6) / thumb.height)
        th = thumb.resize((int(thumb.width * k), int(thumb.height * k)))
        sheet.paste(th, (c * cell + 3, r * (cell + 16) + 3))
        d.text((c * cell + 3, r * (cell + 16) + cell + 2), name, fill=(230, 230, 230), font=font)
    sheet.save(out / "_sheet.png")
    print(f"poses: {len(poses)} -> {out}")
    print(f"sheet: {out / '_sheet.png'}")


if __name__ == "__main__":
    main()
