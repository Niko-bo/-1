# 第二条路线 + K：立绘/差分/事件CG → 动态 CG 的完整实现路径

> **实测进度（2026-11-13）**：阶段 0 ✅ ｜ 阶段 1 ✅ ｜ 阶段 1.5 ✅（97 张候选待筛选）
> 详细数据、踩坑与对照实验：`out/phase_report.md`　审图：`out/qa/gen15/`　待筛清单：`out/gen15/picks.txt`
>
> 机器实测：**RTX 3070 Ti Laptop 8GB / i7-12700H / 16GB RAM / D 盘 514GB 空闲 / Python 3.11**
> 已落地：ComfyUI **0.45.0** + **torch 2.9.1+cu126** + 模型 12GB（`models/`，经 `extra_model_paths.yaml` 挂载）
> 关键实测结论：出图用 **denoise 0.9 + ControlNet 0.85**；**IP-Adapter Plus Face 与 Illustrious 不兼容**（画面发灰，弃用）

路线定义（你选的第二条）：**A 本地 SDXL 动漫底模 + 角色 LoRA + ControlNet** → **E 分层立绘流水线（1488×2272 + 表情差分层）**，最后接 **K 视频**。

---

## 0. 先看这组实测数据（它决定了后面所有参数）

已跑 `scripts/01_prepare_dataset.py` 扫完全部 66 个文件：

| 项目 | 数量 | 说明 |
|---|---|---|
| 唯一立绘底图 | **15** | 原有 17 张，`lob.png == 常装1.png`、`nugi01.png == 常装3.png` 完全同文件 |
| 唯一表情差分 | **11** | 表情夹里 47 个差异文件去重后只有 11 种，在 5 套服装里重复存放 |
| 事件 CG | 1 | `st_04.png`，1200×800，另一套规格 |
| 空占位 | 1 | `non.png` 全透明 |
| 差分统一面部区 | **x 782..928 / y 403..500（146×97）** | 所有差分的并集，这就是重绘遮罩的基准矩形 |

两个直接结论：

1. **表情层里只有眼、眉、嘴、腮红**（见 `out/qa/face_patches.png`），说明你这套系统的差分成本极低——只需要重绘 146×97 那一小块。
2. **15 张底图不够训一个能换姿势的角色 LoRA**，而且 15 张全是同一个站姿。直接训 = 只会复制站姿。所以路径里必须插一个「扩图攒数据集」的阶段（阶段 1.5），这是这条路线唯一容易翻车的地方。

---

## 1. 路径总览

| 阶段 | 做什么 | 产出 | 8GB 耗时 | 验收标准（过不了别进下一阶段） |
|---|---|---|---|---|
| **0** ✅ | 装 ComfyUI + kohya，下模型 | 能文生图 | 已完成（约 1.5 小时） | 832×1216 出图不 OOM |
| **1** ✅ | 数据集整理 | 14 单人立绘 + 32 混合 + 18 差分 | 已完成 | `out/report.md` 数字对得上 |
| **1.5** ✅ | **扩图**：ControlNet 姿势 + 逐图服装提示词 | 97 张候选（7 套 × 12 姿势） | 已完成（约 70 分钟） | 至少 8 个不同姿势/角度，脸是同一张脸 |
| **2** | 训角色 LoRA | `nmk_char_sdxl` + `nmk_char_sd15` | 0.5~1.5 小时/个 | 换姿势不掉脸、披风腰带细节稳定 |
| **3** | 批量出立绘/事件 CG（LoRA+ControlNet+分层） | 底图 α + 分层 | ~1 分钟/张 | 1488×2272 对位正确、无接缝 |
| **4** | 表情差分批量重绘 | 每个底图 × 12 表情 | ~20 秒/张 | 叠回底图无边界、坐标在规范矩形内 |
| **5** | 事件 CG 精修交付 | 1920×1080 / 1200×800 | 半天 | 放大后线稿不糊、光影统一 |
| **6** | **K 视频**（三条子路线按需） | 5s 动态镜头/动态立绘 | 见下 | 首帧=你自己画的 CG，人物不漂 |

预估：**阶段 0~4 约 1.5 周**（大头在阶段 1.5 的筛选），阶段 6 另算。

---

## 2. 各阶段怎么做（含命令）

### 阶段 0：环境（已完成，含实测修正）

```powershell
# 实际使用的安装脚本（已跑通，含镜像与代理绕过）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\00_install_comfyui.ps1 -Root 'D:\dp玩\cg-pipeline'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\01_start_comfy.ps1   -Root 'D:\dp玩\cg-pipeline'

# 模型（hf-mirror，断点续传）
python scripts\04_download_models.py --only core
```

**实测与原计划的三个偏差**（照原文装会卡住）：

1. **torch 必须 ≥2.9**：`comfy_kitchen 0.2.41` 用到 `torch.library.custom_op` 的 `list[int]` 参数，
   torch 2.6 直接 `ValueError: Parameter stride has unsupported type list[int]`。已改用 **2.9.1+cu126**。
2. **git 全局代理指向死端口** `127.0.0.1:10808`：所有 `git` 调用必须带 `-c http.proxy= -c https.proxy=`。
3. **官方 pip 源只有约 50KB/s**：torch 用 `scripts\dl_parallel.py` 多线程分段下（12 线程 ≈10MB/s），
   模型走 `hf-mirror.com`（≈10MB/s）。GitHub 走 `ghproxy.net`。

启动参数（8GB 实测稳定）：`--lowvram --reserve-vram 0.8 --disable-smart-memory --preview-method none`

自定义节点：`ComfyUI_IPAdapter_plus` 已装但**实测弃用**（见阶段 1.5）；后续按需加 `comfyui-rembg`、
`ComfyUI-VideoHelperSuite`、`ComfyUI-FramePackWrapper`、`ComfyUI-GGUF`、`ComfyUI-AnimateDiff-Evolved`。

kohya 训练环境单独建 venv（避免和 ComfyUI 的 torch 打架）：
```powershell
cd D:\dp玩\cg-pipeline
git -c http.proxy= -c https.proxy= clone https://ghproxy.net/https://github.com/kohya-ss/sd-scripts.git
python -m venv envs\kohya
envs\kohya\Scripts\activate
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r sd-scripts\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 阶段 1：数据集（已完成）

```powershell
python scripts\01_prepare_dataset.py
```
产出（新数据集 151 个文件的实测结果）：`out/dataset/char/` **14 张**单人立绘、`out/dataset/char_plus/` **32 张**
（+双人立绘+事件CG+16 张场景图）、`out/dataset/faces/` **18 张**唯一差分、`out/dataset/pairs.json`（三种画布的精确坐标）、
`out/report.md`、`out/dataset/_review.csv`（待人工复核的打标）。

**打标要人工过一遍**：打开 `out/dataset/char/_review.csv`，把每个文件的 `caption` 改准。特别是 `hukumomi_* / hukusimi_*` 这两组我按文件名推不出服装，标的是占位。触发词统一用 `nmk_char`，别改。

### 阶段 1.5：扩图（进行中，配方已实测锁定）

```powershell
python scripts\05_gen_poses.py          # 12 个 OpenPose 18 点骨架
python scripts\08_make_face_refs.py     # 头部参考图（IP-Adapter 备用，本期未采用）
python scripts\06_run_phase15.py --denoise 0.9 --per-pose 1 --face-ref none --cn-strength 0.85
python scripts\07_make_gen15_sheet.py   # 审图表 + 自动废图标记
```

实测确定的配方：**init=对应底图，prompt=底图自己的 caption（剔掉姿势词）+ 目标姿势词，denoise 0.9，ControlNet 0.85，steps 28，cfg 5.5**。
14 底图 × 12 姿势 = 168 张候选，约 36 秒/张。

**四组对照实验（决定了上面的参数）**

| 配置 | 结果 |
|---|---|
| denoise 0.55 | 姿势几乎不变（原图潜变量压住），等于白跑 |
| denoise 0.75 | 半跟随（坐着像靠在椅子前） |
| **denoise 0.90** | **姿势跟随 + 服装来自底图 + 对比度正常 → 采用** |
| denoise 1.00 | 姿势完美，但初始图完全失效，所有底图会出同一套衣服 |
| IP-Adapter Plus Face 0.45 / 0.55 / 0.85 | 画面全部发灰、对比度塌陷，**与 Illustrious 不兼容，弃用** |

另一个坑：提示词曾硬编码 `standing`，与坐/跪姿势冲突，必须删掉，姿势词只由 `pose_tags` 提供。

验收：**至少 8 个不同姿势或角度**、脸一眼是同一人、披风扣和腰带位置对。这一步省了，LoRA 只会画站姿。
若身份还原不够，补一条「脸部移植」路线：把原立绘的脸贴到生成图上，再做 denoise 0.35 的局部重绘（与阶段 4 同一套局部重绘机制）。

### 阶段 2：训 LoRA

```powershell
cd D:\dp玩\cg-pipeline\sd-scripts
..\envs\kohya\Scripts\activate

# SDXL 角色 LoRA（正式出图用）
accelerate launch --num_cpu_threads_per_process 4 sdxl_train_network.py ^
  --config_file D:\dp玩\cg-pipeline\workflows\lora_sdxl_8gb.toml

# SD1.5 角色 LoRA（只给 AnimateDiff 视频用，很便宜）
accelerate launch --num_cpu_threads_per_process 4 train_network.py ^
  --config_file D:\dp玩\cg-pipeline\workflows\lora_sd15_8gb.toml
```
配置里已经为你定好的关键项：`832×1216` 桶、`network_dim 32`、`AdamW8bit`、`gradient_checkpointing`、`cache_latents_to_disk`、`lowram`、**关掉 flip_aug**（你的披风/腰带左右不对称，镜像会让模型学错）。

选权重：每 2 epoch 存一个，训练完用 `sample_prompts.txt` 各跑一遍，挑「换姿势还像」的那个 epoch。通常在第 8~12 epoch。

### 阶段 3：出图 + 分层导出

按 `workflows/comfyui_graphs.md` 的**图 1** 搭：LoRA(0.85) + ControlNet(DW-Pose 0.8) + IP-Adapter(0.6) → 出 832×1216 → 4x 放大 → rembg 扣透明 → 归一化到 1488×2272。

```powershell
python scripts\02_compose_sprite.py --base out\gen\case01_raw.png --out out\gen\case01_norm.png
```

### 阶段 4：表情差分（成本最低、见效最快）

```powershell
python scripts\03_make_inpaint_jobs.py            # 生成 15 底图 × 12 表情 = 180 个任务
# ComfyUI 按 图 2 跑 jobs.jsonl，denoise 用每条的 strength
python scripts\02_compose_sprite.py --base ... --faces ... --outdir out\diffs --sheet --zoom 4
```
遮罩固定 `out/inpaint/masks/face_mask_1488x2272.png`（面部矩形外扩 14px + 3px 羽化）。12 个预设：smile / smile_open / serious / surprised / angry / sad / cry / shy / half_closed / closed_eyes / blank / pant，提示词在 `out/inpaint/prompt_card.md`，自己改词即可扩到 30+ 表情。

**关键**：denoise 低于 0.45 表情不动，高于 0.75 画风开始跑。0.5~0.65 是甜点。

### 阶段 5：事件 CG

事件 CG 和立绘是两套规格，分开出：
- 立绘：832×1216 → 1488×2272（带 α）
- 事件 CG：1216×832 → 4x → 1920×1080（或按你的 `st_04` 用 1200×800）
- 出 CG 时 ControlNet 换成 **Depth**（锁构图与遮挡），LoRA 权重降到 0.7，靠提示词补场景光效
- 后期：统一加一层胶片颗粒 + 轻微色散，让 AI 图和手绘立绘的质感拉齐

---

## 3. K 视频：三条子路线（建议 3a 为主）

| 子路线 | 显存 | 5 秒耗时 | 风格贴合 | 适合 |
|---|---|---|---|---|
| **3a 首尾帧（FramePack）** | 6GB 可跑 | 6~12 分钟 | ●●● 首尾帧都是你的 LoRA 出的图，风格不会跑 | **CG 镜头、特写、转场** |
| **3b AnimateDiff 短循环** | 8GB 轻松 | 2~4 分钟 | ●● 受限于 SD1.5 | 呼吸、发丝、氛围循环、动态立绘底 |
| **3c Wan 2.2 5B GGUF** | 8GB（Q4+lowvram） | 10~20 分钟 | ●● 首帧继承风格，脸会轻微漂 | 有镜头运动的长镜头 |
| 3d 拆层 2.5D / Live2D | 不吃显卡 | — | ●●● 零闪烁 | 游戏内动态立绘（正统做法） |

**推荐组合**：
1. 用**图 1 + `nmk_char_sdxl`** 出首帧 A、尾帧 B（都是你自己的画风，1488×2272 或 1920×1080）
2. **3a FramePack** 只补中间运动 → 5 秒镜头
3. **RIFE 补帧到 30fps** → 剪映加音效/字幕/转场（本机已装剪映）
4. 需要更长的镜头或 720p，就上云端跑 Wan 14B（RunPod 4090 约 ¥2~4/小时），**首尾帧仍然用你本地 LoRA 出的图**——这样风格始终由你控制，视频模型只负责"动"。

如果目标是**游戏内的动态立绘**（呼吸+眨眼+发丝），别用视频模型，走 **3d 拆层 + Live2D**：拆 8~12 个部件（前发/后发/眼白/瞳孔/眉/嘴/披风/身体），Live2D 绑定后零闪烁、可交互，还能直接吃你现有的 11 个表情差分。

---

## 4. 目录与文件

```
D:\dp玩\cg-pipeline\
├─ README.md                       ← 本文件（实现路径）
├─ config\paths.json               参考图/输出路径、触发词、桶尺寸
├─ config\models.json              阶段 0 要下的模型清单与用途
├─ scripts\01_prepare_dataset.py   数据集整理（已跑通）
├─ scripts\02_compose_sprite.py    底图+差分层合成、坐标校验、审图sheet（已跑通）
├─ scripts\03_make_inpaint_jobs.py 生成 180 个表情重绘任务（已跑通）
├─ workflows\lora_sdxl_8gb.toml    SDXL 训练配置（8GB 专用）
├─ workflows\lora_sd15_8gb.toml    SD1.5 训练配置（给视频用）
├─ workflows\comfyui_graphs.md     三套工作流节点规格
├─ workflows\sample_prompts.txt    训练过程采样提示词
└─ out\                            数据集、遮罩、任务单、审图
```

## 5. 现在就能验证的两张图

- `out\qa\face_patches.png` —— 11 个唯一表情排在一起（证明只有眼眉嘴腮红）
- `out\qa\compose_test\sheet.png` + `*_facezoom.png` —— 差分层叠回 `常装1` 的效果，3 倍放大看边界

## 6. 建议的下一步

按投入产出比，推荐这个顺序：

1. **先跑阶段 0**（装环境 + 下模型，2~3 小时，无人值守）——我可以把下载脚本和 ComfyUI 启动脚本直接写好
2. **同时做阶段 4 的表情差分**——你现在就能用现成 15 张底图刷出 180 个表情，不需要等 LoRA
3. 阶段 1.5 扩图 → 阶段 2 训 LoRA（这一步开始才真正需要显卡排队）

要不要我现在先把 **ComfyUI 自动安装 + 模型下载脚本** 和 **图 1/图 2 的 workflow JSON** 生成出来？装完你拖进去就能跑。
