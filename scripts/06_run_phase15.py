"""Phase 1.5-b: 批量生成「同人设 + 新姿势」候选图（ComfyUI API）。

策略（不依赖任何自定义节点，先保证能出图）：
  数据集立绘(832x1216) --VAEEncode--> KSampler(denoise 0.5~0.65)
  骨架图 --ControlNet(openpose, 0.85)--> 锁姿势
  提示词带全量角色特征 tag --> 保持发色/瞳色/服装
这样人物来自你自己的底图潜变量，姿势来自骨架，两头都可控。

用法：
  python scripts/06_run_phase15.py --poses out/poses --bases out/dataset/char \
      --per-pose 2 --host 127.0.0.1:8188
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

POS = ("masterpiece, best quality, amazing quality, very aesthetic, absurdres, "
       "{char}, {pose_tags}, solo, 1girl, silver hair, very long hair, hair over one eye, "
       "blue eyes, mature female, large breasts, tall, {outfit}, "
       "simple background, white background")

# caption 里描述姿势的词会和目标骨架冲突，逐底图复用 caption 时必须先剔掉
POSE_WORDS = {
    "standing", "full body", "looking at viewer", "solo", "upper body", "cowboy shot",
    "from behind", "from side", "sitting", "kneeling", "indoors", "wooden wall",
    "cinematic lighting", "multiple boys", "faceless male", "adult", "solo focus",
    "against wall", "grabbed from behind", "crowded", "depth of field", "1boy",
}


def outfit_from_caption(base: Path) -> str:
    """从底图的 .txt 打标里取服装特征（去掉姿势/场景/人物数量词）。"""
    txt = base.with_suffix(".txt")
    if not txt.exists():
        return "white dress shirt, black pencil skirt, black pantyhose"
    tags = [t.strip() for t in txt.read_text(encoding="utf-8").split(",")]
    keep = [t for t in tags if t and t.lower() not in POSE_WORDS
            and t not in ("nmk_char", "1girl", "silver hair", "very long hair",
                          "hair over one eye", "blue eyes", "mature female",
                          "large breasts", "tall", "detailed eyes")]
    return ", ".join(keep) if keep else "white dress shirt, black pencil skirt"
NEG = ("lowres, worst quality, low quality, bad anatomy, bad hands, extra digits, "
       "fewer digits, deformed, mutated, ugly, blurry, jpeg artifacts, watermark, signature, "
       "text, multiple views, reference sheet, extra limbs, bad proportions, "
       "male focus, 1boy, cropped head, out of frame, "
       "dress, gown, wedding dress, long skirt, short hair, blue hair, twintails, "
       "different character, wrong eye color, wrong hair color")


def build_graph(ckpt_name: str, cn_name: str, init_name: str, pose_name: str,
                pos: str, neg: str, seed: int, denoise: float,
                cn_strength: float, steps: int, cfg: float, filename_prefix: str,
                face_ref: str | None = None, ip_weight: float = 0.8,
                ip_file: str = "", clip_name: str = "",
                weight_type: str = "linear", embeds_scaling: str = "V only"):
    """ComfyUI API 格式（节点 id -> class_type/inputs）。给了 face_ref 就挂 IP-Adapter 锁脸。"""
    model_ref = ["1", 0]
    g = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": pos, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": neg, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": pose_name}},
        "5": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cn_name}},
        "6": {"class_type": "ControlNetApplyAdvanced",
              "inputs": {"positive": ["2", 0], "negative": ["3", 0],
                         "control_net": ["5", 0], "image": ["4", 0],
                         "strength": cn_strength, "start_percent": 0.0, "end_percent": 0.9}},
        "7": {"class_type": "LoadImage", "inputs": {"image": init_name}},
        "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": model_ref, "positive": ["6", 0], "negative": ["6", 1],
                         "latent_image": ["8", 0], "seed": seed, "steps": steps,
                         "cfg": cfg, "sampler_name": "dpmpp_2m_sde",
                         "scheduler": "karras", "denoise": denoise}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": filename_prefix}},
    }
    if face_ref:
        g["12"] = {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": ip_file}}
        g["13"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_name}}
        g["14"] = {"class_type": "LoadImage", "inputs": {"image": face_ref}}
        g["15"] = {"class_type": "IPAdapterAdvanced",
                   "inputs": {"model": ["1", 0], "ipadapter": ["12", 0], "image": ["14", 0],
                              "clip_vision": ["13", 0], "weight": ip_weight,
                              "weight_type": weight_type, "combine_embeds": "concat",
                              "start_at": 0.0, "end_at": 1.0, "embeds_scaling": embeds_scaling}}
        g["9"]["inputs"]["model"] = ["15", 0]
    return g


def post_json(host: str, path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://{host}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def get_json(host: str, path: str) -> dict:
    with urllib.request.urlopen(f"http://{host}{path}", timeout=60) as r:
        return json.loads(r.read())


def wait_done(host: str, pid: str, timeout: float = 1800) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = get_json(host, f"/history/{pid}")
        except urllib.error.URLError:
            time.sleep(2)
            continue
        if pid in h:
            return h[pid]
        time.sleep(2)
    raise TimeoutError(f"job {pid} timeout")


def fetch_image(host: str, item: dict, dest: Path) -> bool:
    q = urllib.parse.urlencode({"filename": item["filename"],
                                "subfolder": item.get("subfolder", ""),
                                "type": item.get("type", "output")})
    with urllib.request.urlopen(f"http://{host}/view?{q}", timeout=120) as r:
        data = r.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return len(data) > 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--ckpt", default="illustriousXL_v01.safetensors")
    ap.add_argument("--cn", default="controlnet-openpose-sdxl.safetensors")
    ap.add_argument("--bases", default=str(ROOT / "out" / "dataset" / "char"))
    ap.add_argument("--poses", default=str(ROOT / "out" / "poses"))
    ap.add_argument("--out", default=str(ROOT / "out" / "gen15"))
    ap.add_argument("--comfy-input", default=str(ROOT / "ComfyUI" / "input"))
    ap.add_argument("--per-pose", type=int, default=2)
    ap.add_argument("--denoise", type=float, nargs="+", default=[0.55, 0.65])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=5.5)
    ap.add_argument("--cn-strength", type=float, default=0.85)
    ap.add_argument("--char", default="nmk_char")
    ap.add_argument("--limit-bases", type=int, default=0)
    ap.add_argument("--base-filter", default=None, help="只跑文件名包含这些子串的底图（逗号分隔）")
    ap.add_argument("--force-outfit", default=None, help="覆盖所有底图的服装提示词（补跑用）")
    ap.add_argument("--pose-filter", default=None, help="只跑文件名包含该子串的姿势")
    ap.add_argument("--face-ref", default=str(ROOT / "out" / "ipadapter" / "face_normal.png"),
                    help="IP-Adapter 头部参考图；给 none 则不挂 IP-Adapter")
    ap.add_argument("--ip-weight", type=float, default=0.8)
    ap.add_argument("--weight-type", default="linear")
    ap.add_argument("--embeds-scaling", default="V only")
    ap.add_argument("--ip-file", default="ip-adapter-plus-face_sdxl_vit-h.safetensors")
    ap.add_argument("--clip-name", default="CLIP-ViT-H-14.safetensors")
    ap.add_argument("--no-feed", action="store_true", help="不复制素材（调试用）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cinput = Path(args.comfy_input)
    cinput.mkdir(parents=True, exist_ok=True)

    bases = sorted(Path(args.bases).glob("*.png"))
    if args.base_filter:
        keys = [k.strip() for k in args.base_filter.split(",") if k.strip()]
        bases = [b for b in bases if any(k in b.stem for k in keys)]
    if args.limit_bases:
        bases = bases[:args.limit_bases]
    poses = sorted(p for p in Path(args.poses).glob("*.png") if not p.name.startswith("_"))
    if args.pose_filter:
        keys = [k.strip() for k in args.pose_filter.split(",") if k.strip()]
        poses = [p for p in poses if any(k in p.stem for k in keys)]
    if not bases or not poses:
        print(f"no inputs: bases={len(bases)} poses={len(poses)}")
        return 2

    # LoadImage 只能读 ComfyUI/input，先把素材投喂进去（幂等）
    def feed(src: Path, prefix: str) -> str:
        dst = cinput / f"{prefix}_{src.name}"
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            dst.write_bytes(src.read_bytes())
        return dst.name

    base_names = {b: feed(b, "init") for b in bases}
    pose_names = {p: feed(p, "pose") for p in poses}
    face_ref = None
    if args.face_ref and args.face_ref.lower() != "none":
        fp = Path(args.face_ref)
        if not fp.exists():
            print(f"face ref missing: {fp}")
            return 2
        face_ref = feed(fp, "face")

    # 姿势 -> 额外提示词（帮模型理解俯仰/朝向）
    pose_tags = {
        "01_standing_front": "standing, facing viewer, arms at sides",
        "02_three_quarter_hip": "standing, hand on hip, cowboy shot",
        "03_hands_behind": "standing, hands behind back",
        "04_sitting_chair": "sitting, on chair, knees together",
        "05_sitting_floor": "sitting, hugging own knees, on floor",
        "06_kneeling": "kneeling",
        "07_leaning_wall": "leaning back, against wall, one hand on wall",
        "08_arms_crossed": "standing, arms crossed, under chest, confident",
        "09_hand_on_chest": "standing, hand on own chest, looking at viewer",
        "10_waist_up": "cowboy shot, hand on hip, looking at viewer",
        "11_looking_back": "looking back, over shoulder",
        "12_arms_up": "standing, arms up, stretching",
    }

    jobs = []
    rnd = random.Random(20261113)
    for b in bases:
        for p in poses:
            for k in range(args.per_pose):
                den = args.denoise[k % len(args.denoise)]
                seed = rnd.randrange(1, 2 ** 31)
                name = f"{b.stem}__{p.stem}__d{int(den*100)}__s{seed}"
                jobs.append({"base": b, "pose": p, "denoise": den, "seed": seed, "name": name})

    outfits = {b: (args.force_outfit or outfit_from_caption(b)) for b in bases}
    print(f"jobs: {len(jobs)}  (bases={len(bases)} x poses={len(poses)} x {args.per_pose})")
    manifest = []
    ok = 0
    for i, j in enumerate(jobs, 1):
        pos = POS.format(char=args.char, pose_tags=pose_tags.get(j["pose"].stem, "standing"),
                         outfit=outfits[j["base"]])
        g = build_graph(args.ckpt, args.cn, base_names[j["base"]], pose_names[j["pose"]],
                        pos, NEG, j["seed"], j["denoise"], args.cn_strength, args.steps,
                        args.cfg, f"phase15/{j['name']}", face_ref=face_ref,
                        ip_weight=args.ip_weight, ip_file=args.ip_file,
                        clip_name=args.clip_name, weight_type=args.weight_type,
                        embeds_scaling=args.embeds_scaling)
        try:
            r = post_json(args.host, "/prompt", {"prompt": g, "client_id": str(uuid.uuid4())})
            pid = r["prompt_id"]
            hist = wait_done(args.host, pid)
            imgs = [im for node in hist.get("outputs", {}).values()
                    for im in node.get("images", [])]
            if not imgs:
                print(f"[{i}/{len(jobs)}] {j['name']} NO IMAGE")
                continue
            dest = out / f"{j['name']}.png"
            fetch_image(args.host, imgs[0], dest)
            ok += 1
            manifest.append({"file": str(dest), "base": str(j["base"]), "pose": j["pose"].stem,
                             "denoise": j["denoise"], "seed": j["seed"]})
            print(f"[{i}/{len(jobs)}] {dest.name} OK", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(jobs)}] {j['name']} FAIL {type(e).__name__}: {e}", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndone {ok}/{len(jobs)} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    import urllib.parse  # noqa: E402  (used in fetch_image)
    raise SystemExit(main())
