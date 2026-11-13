"""Phase 1.5-a2: 从立绘裁出「头部参考图」，供 IP-Adapter Plus Face 锁身份。

1488x2272 画布上实测面部区是 x782..928 / y403..500（只含眼眉嘴），
IP-Adapter 需要包含头发与脸型的头部区域，所以按脸中心外扩取方形。

产出：out/ipadapter/face_<name>.png（512x512 白底）与 _sheet.png
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(r"D:\浏览器\新人物设计\新建文件夹\人物初版设计图")
FACE_C = (855, 451)     # 面部区中心（1488x2272 画布）
HALF = 210              # 半边长 -> 420x420 头部框
OUT_SIZE = 512

EXTRA = {
    "hukumomi": Path(r"D:\浏览器\新人物设计\新建文件夹\表情\hukumomi.png"),
    "normal": Path(r"D:\浏览器\新人物设计\新建文件夹\表情\normal.png"),
}


def main() -> int:
    out = ROOT / "out" / "ipadapter"
    out.mkdir(parents=True, exist_ok=True)

    sources = {p.stem: p for p in SRC.glob("*.png")}
    sources.update(EXTRA)

    made = []
    for name, path in sorted(sources.items()):
        img = Image.open(path).convert("RGBA")
        if img.size == (1200, 800) or img.width != 1488:
            print(f"skip {name} ({img.width}x{img.height})")
            continue
        box = (FACE_C[0] - HALF, FACE_C[1] - HALF, FACE_C[0] + HALF, FACE_C[1] + HALF)
        crop = img.crop(box)
        bg = Image.new("RGB", crop.size, (255, 255, 255))
        bg.paste(crop, (0, 0), crop)
        bg = bg.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
        dst = out / f"face_{name}.png"
        bg.save(dst)
        made.append((name, dst))
        print(f"{name} -> {dst.name}")

    if made:
        cols = 5
        cell = 180
        rows = (len(made) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rows * (cell + 16)), (235, 235, 238))
        d = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except Exception:
            font = ImageFont.load_default()
        for i, (name, path) in enumerate(made):
            r, c = divmod(i, cols)
            th = Image.open(path).resize((cell - 8, cell - 8), Image.LANCZOS)
            sheet.paste(th, (c * cell + 4, r * (cell + 16) + 4))
            d.text((c * cell + 4, r * (cell + 16) + cell + 2), name, fill=(20, 20, 20), font=font)
        sheet.save(out / "_sheet.png")
        print(f"sheet -> {out / '_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
