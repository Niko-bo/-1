# 阶段 0 / 1 / 1.5 实测报告

时间：2026-11-13　机器：RTX 3070 Ti Laptop 8GB / i7-12700H / 16GB

## 一、阶段 0：环境（已完成）

| 项 | 结果 |
|---|---|
| ComfyUI | **0.45.0**，`D:\dp玩\cg-pipeline\ComfyUI` |
| Python | 独立 venv：`envs\comfy`（Python 3.11.6） |
| PyTorch | **2.9.1+cu126**（CUDA 可用，sm_86） |
| torchvision | 0.24.1+cu126 |
| 启动参数 | `--lowvram --reserve-vram 0.8 --disable-smart-memory --preview-method none` |
| 模型目录 | `D:\dp玩\cg-pipeline\models`（由 `ComfyUI\extra_model_paths.yaml` 挂载） |
| 自定义节点 | `ComfyUI_IPAdapter_plus`（实测后弃用，见下） |

已下载模型（约 12GB）：

| 文件 | 体积 | 用途 |
|---|---|---|
| `illustriousXL_v01.safetensors` | 6.46GB | 主底模（动漫/SDXL） |
| `controlnet-openpose-sdxl.safetensors` | 2.33GB | 姿势锁定 |
| `CLIP-ViT-H-14.safetensors` | 2.35GB | IP-Adapter 的视觉编码器 |
| `ip-adapter-plus-face_sdxl_vit-h.safetensors` | 0.79GB | 锁脸（未采用） |
| `4x-AnimeSharp.pth` | 0.06GB | 后续放大 |

**三个必须记下来的坑**

1. **torch 2.6 装不上**：`comfy_kitchen 0.2.41` 用了 `torch.library.custom_op` 的 `list[int]` 参数，
   torch 2.6 的 `infer_schema` 不支持，启动即 `ValueError: Parameter stride has unsupported type list[int]`。
   升到 **2.9.1** 后正常。
2. **git 全局代理是死的**：`http.proxy=127.0.0.1:10808` 未运行，所有 git 操作直接失败。
   脚本里统一用 `git -c http.proxy= -c https.proxy=` 绕过。
3. **官方 pip 源被限速到 50KB/s**：torch wheel 改用自写的多线程分段下载（`scripts/dl_parallel.py`，12 线程 ≈10MB/s），
   模型走 `hf-mirror.com`（≈10MB/s）。

## 二、阶段 1：数据集（已完成）

新目录 `D:\浏览器\新人物设计\新建文件夹`，151 个 PNG / 4 个子目录。

| 类别 | 数量 | 去向 |
|---|---|---|
| `sprite` 单人立绘 | **14** | `out/dataset/char/`（身份训练主力） |
| `sprite_multi` 双人立绘 | 1（`hukumomi`） | `out/dataset/char_plus/` |
| `event_cg` 事件 CG | 1（`st_04`） | `char_plus` |
| `pose_cg` 场景图 | 16 | `char_plus`（补姿势） |
| `face_patch` 表情差分 | 54（去重后 **18**） | `out/dataset/faces/` |
| `storyboard` 分镜 CG | 54 | 仅参考，不进训练 |
| `layer` 局部件（手/遮罩） | 8 | 排除 |
| `empty` | 1（`non.png`） | 排除 |

- `out/dataset/char_plus/` = **32 张**（14 单人 + 1 双人 + 1 事件 CG + 16 场景）
- 差分画布有三种规格：`1488x2272`（11 张）、`868x875`（6 张）、`1200x800`（1 张），
  合成时必须按画布分组，不能混用坐标。
- 打标清单：`out/dataset/_review.csv`，`REVIEW_OUTFIT` / `NSFW_SCENE` 两列标记需要人工复核的条目。

**修复的两个数据问题**

- 立绘 alpha 边缘有极淡噪点，`getbbox()` 会把包围盒撑到画布边缘（导致误判双人）。改用阈值 96 的实心包围盒。
- `hukumomi.png` 是双人图，最初被当单人流进 `char/` 且 caption 写 `solo`——会让 LoRA 学错。已按包围盒宽度 >45% 画布识别并移入 `char_plus`。

## 三、阶段 1.5：扩图（进行中）

姿势来源：程序生成 **12 个 OpenPose 18 点骨架**（`out/poses/`），
比从成品插画里跑 DW-Pose 干净（不会把旁边的第二角色一起检出）。

出图配方（实测确定）：

```
illustriousXL_v01 + controlnet-openpose-sdxl
  init = 对应底图（832x1216，来自 char/）
  prompt = 底图自己的 caption（去掉姿势词）+ 目标姿势词 + 角色特征
  denoise = 0.9    CN strength = 0.85    steps 28    cfg 5.5
```

**四组对照实验（决定了上面的参数）**

| 配置 | 结果 |
|---|---|
| denoise 0.55 | 姿势几乎不变（原图潜变量压住），等于白跑 |
| denoise 0.75 | 半跟随（坐着像靠在椅子前） |
| **denoise 0.90** | **姿势跟随 + 服装来自底图 + 对比度正常 → 采用** |
| denoise 1.00 | 姿势完美，但初始图完全失效，所有底图会出同一套衣服 |
| IP-Adapter Plus Face 0.45 / 0.55 / 0.85 | 画面全部发灰、对比度塌陷，**与 Illustrious 不兼容，弃用** |
| `K+V` 缩放 + 0.55 | 仍然发灰，确认不是权重问题 |

**姿势库两轮修正**（第一版有三个系统性缺陷，跑完 26 张后停机重做）

| 姿势 | 第一版问题 | 处理 |
|---|---|---|
| 背面 08 | 骨架仍画了鼻/眼关键点 → ControlNet 输出了 3/4 侧脸 | 删除（对话立绘用不上背面） |
| 侧面 09 | 骨架四肢重叠成一条竖线，CN 几乎无信息 | 删除 |
| 胸像 10 | 腿画到画布外，骨架仍填满画面 → 还是全身 | 改为 `10_waist_up`（cowboy shot，无腿线），实测成立 |
| 新增 08/09 | — | `08_arms_crossed`（抱臂）、`09_hand_on_chest`（抚胸），实测成立 |

另一个坑：`huku*` 两组我按文件名猜成了「制服 + 百褶裙」，实际是白衬衫 + 黑裙，
导致 hukusimi 生成的图穿上了带红领结的水手服。已修正为常装同一套标签。

> 提示词里曾硬编码 `standing`，和坐/跪姿势冲突，已删除；现在姿势词由 `pose_tags` 单独提供。

产量：7 套底图 × 12 姿势 = **97 张候选**（`out/gen15`，含少量额外种子），
自动标记失败 14 张（`cropped` 为主，多出现在「举手/坐地」这类头发越界的姿势）。

审图：`out/qa/gen15/<底图>.png`（每张表第一格是原立绘做对照，红框=自动标记）；
机器可读指标：`out/qa/gen15/gen15_flags.csv`；
待筛清单模板：`out/gen15/picks.txt`（删掉不要的行即可）。

**运行层面的一个坑**：`--disable-smart-memory` 去掉后单张从 75s 降到 ~40s，
但连续跑 60 张后出现显存/内存碎片化（某张图退化到 181s/步）。
对策：**每 24~30 张重启一次 ComfyUI**，或把该参数加回来换取稳定（慢一倍）。

## 四、阶段 1.5 的产出质量（97 张抽检结论）

| 维度 | 状态 |
|---|---|
| 姿势跟随 | 好：站/坐椅/坐地/跪/靠墙/抱臂/抚胸/叉腰/回头/举手 都能稳定落到骨架上 |
| 服装还原 | 好：白衬衫+黑裙+披风/内衣 等主要特征保留；`nugi*`（脱衣阶段）会自行发明长裙，需补跑 |
| 头发 | 中：银发保留，但长度在少数图里变短（已加 `short hair` 负词，尚未回炉） |
| 脸 | 中：属于底模的默认脸，与参考立绘有差距；这批图用于补姿势，身份仍靠 14 张原始立绘兜底 |
| 手部 | 中：抱臂/抚胸/扶墙偶尔崩，需人工筛 |

## 五、下一步（阶段 1.5 收尾 → 阶段 2）

1. 打开 `out/qa/gen15/*.png`，从 97 张里挑 25~40 张，把名字留在 `out/gen15/picks.txt`
2. `python scripts\09_curate.py --picks out\gen15\picks.txt` → 生成 `out/dataset/final/`（含打标）
3. 进阶段 2：按 `workflows/lora_sdxl_8gb.toml` 训 SDXL 角色 LoRA（8GB 约 0.5~1.5 小时）
4. 可选补跑：`nugi02 / nugi04` 两组用 `--force-outfit "black lace lingerie, open white shirt, black pantyhose"` 重跑一遍
5. 若身份还原不够，再补「脸部移植」：原脸贴到生成图 → denoise 0.35 局部重绘（与阶段 4 同一套机制）
