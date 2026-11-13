# ComfyUI 三套工作流搭建规格（8GB 显存版）

安装好 ComfyUI 后按下面的节点顺序连线；或者等我把这些直接导出成 `.json` 拖进去。
所有节点名都是 ComfyUI 核心节点名，自定义节点另标注。

启动参数（8GB 必须）：
```
python main.py --lowvram --reserve-vram 0.8 --use-sage-attention --disable-smart-memory
```

---

## 图 1：SDXL 立绘 / 事件 CG 出图（主工作流）

```
[CheckpointLoaderSimple  sdxl_anime_base]
   ├─ MODEL ─→ [LoraLoaderModelOnly  nmk_char_sdxl  strength 0.85]
   │              └─ MODEL ─→ [ControlNetApplyAdvanced] ─→ [KSampler]
   ├─ CLIP  ─→ [CLIPTextEncode 正向] ─┐
   ├─ CLIP  ─→ [CLIPTextEncode 负向] ─┤
   └─ VAE   ─→ [VAEDecode] ←──────────┘
[LoadImage  姿势参考 / 空白 1024x1536]
   └─ IMAGE ─→ [DWPreprocessor  (comfyui_controlnet_aux)] ─→ [ControlNetApplyAdvanced]
[ControlNetLoader  controlnet_pose_sdxl] ─→ [ControlNetApplyAdvanced]
[IPAdapterUnifiedLoader  PLUS FACE] ─→ [IPAdapter  weight 0.6]  ← 脸参考图 LoadImage
[EmptyLatentImage  832x1216] ─→ [KSampler]
[KSampler] → [VAEDecode] → [SaveImage]
```

参数起点：
- sampler `dpmpp_2m_sde` / scheduler `karras` / steps 28 / cfg 6.0 / denoise 1.0
- LoRA 权重 0.8~0.9；IP-Adapter 权重 0.55~0.7（太高会锁死脸型、换不了角度）
- 姿势控制 strength 0.7~0.9；要自由构图时 strength 0.4

顺带出 alpha（立绘需要透明底）：
```
[VAEDecode] → [ImageRemBG (comfyui-rembg)] → [SaveImage RGBA]
```
粗修毛边交给 `scripts/02_compose_sprite.py --zoom 4` 出的放大图人工过一遍。

---

## 图 2：表情差分 inpaint（吃 `out/inpaint/jobs.jsonl`）

```
[LoadImage  base.png        ] ─┐
[LoadImage  face_mask_*.png ] ─┴→ [VAEEncodeForInpaint] → [SetLatentNoiseMask]
                                     └→ LATENT ─→ [KSampler denoise=jobs.denoise]
[CheckpointLoaderSimple + LoRA(0.7)] ─→ MODEL/CLIP
[CLIPTextEncode  jobs.positive / jobs.negative]
[KSampler] → [VAEDecode] → [ImageCompositeMasked base 回贴] → [SaveImage]
```

要点：
- 遮罩固定 `out/inpaint/masks/face_mask_1488x2272.png`（面部矩形外扩 14px + 3px 羽化）
- denoise 0.5~0.7：低了表情不动，高了画风跑偏
- 出图后必须 `02_compose_sprite.py --zoom 4` 验证边界有没有接缝
- 批量跑用 ComfyUI 的 API 模式：`python scripts/04_run_jobs.py`（需要时我写好）

---

## 图 3：视频（K 路线，三条按需选）

### 3a 首尾帧驱动（推荐，风格最稳）
用图 1 + `nmk_char_sdxl` 出首帧 A 和尾帧 B（两边都是你自己的画风），只让模型补中间运动。
```
[LoadImage  frame_A] [LoadImage  frame_B]
[FramePackSampler (ComfyUI-FramePackWrapper)]  total_second_length=5, latent_window_size=9
   → [VHS_VideoCombine (ComfyUI-VideoHelperSuite)] fps 30
```
8GB 显存可跑（FramePack 官方目标 6GB），5 秒 480p ≈ 6~12 分钟。

### 3b AnimateDiff 短循环（最省，适合呼吸/发丝/氛围）
```
[ADE_LoadAnimateDiffModel  mm_sd_v15_v2.ckpt]
[ADE_AnimateDiffLoaderGen1] ← [LoraLoader  nmk_char_sd15  0.8]
[ADE_LoadMotionLoRA  v2_lora_ZoomIn / PanLeft]  → [ADE_AnimateDiffUniformContextOptions  context 16]
[EmptyLatentImage  batch_size 16, 512x768] → [KSampler 25 steps] → [VAEDecode] → VideoCombine
```
16 帧 @8fps，约 2~4 分钟一条；用剪辑软件循环 + 变速。
缺点：只能 SD1.5，所以必须用 3b 专用的 `nmk_char_sd15` LoRA。

### 3c Wan 2.2 TI2V-5B（镜头级运动，8GB 需量化）
```
[UnetLoaderGGUF  wan2.2_ti2v_5B_Q4_K_M.gguf]
[CLIPLoader  umt5_xxl  type=wan]  [VAELoader  wan2.2_vae]
[WanImageToVideo  width 704 height 480 length 49]  ← 首帧=立绘/事件CG
[KSampler 20 steps cfg 5] → [VAEDecode] → [VHS_VideoCombine]
```
8GB + GGUF Q4 + `--lowvram`：480p / 5 秒 ≈ 10~20 分钟。要 720p 或更长就上云端。

### 3d 后期（所有路线共用）
补帧 + 稳定：RIFE / FILM 补到 30fps → 剪映合成（本机已装）→ 加字幕/音效/转场。
