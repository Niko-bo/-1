# 换台电脑继续做：恢复步骤

目标：在新机器上 30~40 分钟内恢复到「能继续出图 / 能继续筛图」的状态。
**模型（12GB）和 ComfyUI 源码不入库**，由脚本重新拉取；仓库里放的是工具、配置、数据集和当前产出。

仓库地址：
- HTTPS：`https://github.com/Niko-bo/-1.git`
- SSH：`git@github.com:Niko-bo/-1.git`（本机 22 端口被墙，用的是 HTTPS + token）

---

## 0. 前置条件

| 项 | 要求 |
|---|---|
| 系统 | Windows 10/11 x64 |
| Python | **3.11**（torch wheel 是 cp311，版本不对装不上） |
| Git | 任意较新版本 |
| 显卡 | NVIDIA，8GB 显存可跑（本项目实测 3070 Ti Laptop 8GB） |
| 网络 | GitHub 需要代理；模型走 `hf-mirror.com` 不需要 |

```powershell
python -c "import sys;print(sys.version)"   # 必须是 3.11.x
```

## 1. 克隆仓库

```powershell
# 有代理时（端口换成你自己的）
git -c http.proxy=http://127.0.0.1:10808 clone https://github.com/Niko-bo/-1.git cg-pipeline

# 中文文件名显示正常（可选）
cd cg-pipeline
git config core.quotepath false
```

> 本机没有 SSH key 也能用：HTTPS 走 Windows 凭据管理器里的 PAT。
> 若要用 SSH，先 `ssh-keygen -t ed25519`，把公钥加到 GitHub，再把 remote 换成 SSH 形式。

## 2. 装 ComfyUI + torch（一条命令）

```powershell
cd cg-pipeline
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\00_install_comfyui.ps1 -Root (Get-Location).Path
```

脚本做的事（都是实测过的路径）：
1. 用 `ghproxy.net` 拉 ComfyUI 源码（直连 github 会超时）
2. 建 `envs\comfy` 虚拟环境
3. 多线程下载 **torch 2.9.1+cu126 / torchvision 0.24.1+cu126**（官方源单连接只有 1MB/s，`dl_parallel.py` 12 线程约 10MB/s）
4. 装 ComfyUI requirements（清华源 + PyPI 兜底）+ `opencv-python-headless`
5. 生成 `ComfyUI\extra_model_paths.yaml`，把模型目录指到仓库外的 `models/`

> **为什么必须是 torch 2.9+**：`comfy_kitchen 0.2.41` 用了 `torch.library.custom_op` 的 `list[int]` 参数，
> torch 2.6 会直接 `ValueError: Parameter stride has unsupported type list[int]`，启动即崩。

## 3. 下载模型（约 12GB）

```powershell
python scripts\04_download_models.py --only core
```

| 文件 | 体积 | 用途 |
|---|---|---|
| `checkpoints/illustriousXL_v01.safetensors` | 6.46GB | 主底模 |
| `controlnet/controlnet-openpose-sdxl.safetensors` | 2.33GB | 姿势控制 |
| `clip_vision/CLIP-ViT-H-14.safetensors` | 2.35GB | IP-Adapter 用（本项目已弃用，可跳过） |
| `ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors` | 0.79GB | 同上 |
| `upscale_models/4x-AnimeSharp.pth` | 0.06GB | 放大 |

只要出图的话，**前两个**（8.8GB）就够。

## 4. 启动服务

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\01_start_comfy.ps1 -Root (Get-Location).Path
# 打开 http://127.0.0.1:8188
```

8GB 显存参数：`--lowvram --reserve-vram 0.8 --preview-method none`。
**注意**：连续生成 >60 张会出现显存碎片化（单张退化到 180s），每 24~30 张重启一次服务。

## 5. 继续出图（阶段 1.5）

```powershell
# 骨架已在仓库里（out/poses/），直接用
python scripts\06_run_phase15.py --denoise 0.9 --per-pose 1 --face-ref none `
    --cn-strength 0.85 --base-filter "常装1" --out out\gen15
python scripts\07_make_gen15_sheet.py     # 出审图表 + 自动废图标记
```

配方（实测锁定）：**denoise 0.9 + ControlNet 0.85 + 逐图服装提示词**，832×1216，28 步，cfg 5.5。

## 6. 继续筛图 → 训 LoRA

```powershell
# 1) 打开 out/qa/gen15/*.png 看对照表，从 out/gen15/picks.txt 里删掉不要的行
# 2) 生成正式训练集（带打标）
python scripts\09_curate.py --picks out\gen15\picks.txt
```

训 LoRA（阶段 2，需另装 kohya）：

```powershell
git -c http.proxy= -c https.proxy= clone https://ghproxy.net/https://github.com/kohya-ss/sd-scripts.git
python -m venv envs\kohya
envs\kohya\Scripts\activate
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r sd-scripts\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
accelerate launch --num_cpu_threads_per_process 4 sdxl_train_network.py --config_file ..\workflows\lora_sdxl_8gb.toml
```

## 7. 路径要改的地方

`config/paths.json` 里是**原数据集**的绝对路径（`D:\浏览器\新人物设计\新建文件夹`）。
换机后如果只是继续出图/训 LoRA，**不用动它**——`out/dataset/` 里已经有整理好的 14+32 张和打标。
只有想从原始素材重新跑 `01_prepare_dataset.py` 时，才需要把 `paths.json` 改成新机器的素材路径。

## 8. 不要提交的东西

`.gitignore` 已排除：`models/`（12GB）、`envs/`（5.5GB）、`wheels/`（4.8GB）、`ComfyUI/`（源码，可重拉）、日志。
这些东西都能用上面的脚本重建，不要塞进 git。
