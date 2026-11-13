# 工程进度状态（2026-11-13）

> 换机恢复步骤见 `SETUP.md`；实测细节与踩坑见 `out/phase_report.md`。

## 一句话状态

阶段 0/1/1.5 已完成，**97 张姿势候选图待人工筛选**，筛完即可进阶段 2（训角色 LoRA）。

## 阶段进度

| 阶段 | 状态 | 产物 | 位置 |
|---|---|---|---|
| 0 环境 | ✅ | ComfyUI 0.45.0 + torch 2.9.1+cu126 + 12GB 模型 | 模型不入库，用 `scripts/04_download_models.py` 重下 |
| 1 数据集 | ✅ | 14 单人立绘 / 32 混合集 / 18 唯一差分 | `out/dataset/{char,char_plus,faces}` |
| 1.5 扩图 | ✅ 待筛选 | 97 张候选（7 套底图 × 12 姿势） | `out/gen15/` + 审图 `out/qa/gen15/` |
| 2 训 LoRA | ⬜ 未开始 | 配置文件已就绪 | `workflows/lora_sdxl_8gb.toml` |
| 3 出图分层 | ⬜ | 合成脚本已就绪 | `scripts/02_compose_sprite.py` |
| 4 表情差分 | ⬜ | 任务生成器已就绪（180 个任务） | `scripts/03_make_inpaint_jobs.py` |
| 5 事件 CG | ⬜ | — | — |
| 6 K 视频 | ⬜ | 方案已定（首尾帧 FramePack 为主） | `workflows/comfyui_graphs.md` |

## 当前可交接的关键结论（省得重新试）

- 出图配方：**denoise 0.9 + ControlNet(openpose) 0.85 + 逐图服装提示词**，832×1216
- **IP-Adapter Plus Face 与 Illustrious 不兼容**（画面发灰），已弃用
- denoise 0.55 姿势不动、1.0 会让所有底图出同一套衣服
- 姿势骨架在 `out/poses/`（12 个，背面/侧面/胸像已验证不可用，已换成抱臂/抚胸/叉腰）
- 连续生成超过 ~60 张会显存碎片化（退化到 180s/步），**每 24~30 张重启一次 ComfyUI**
- 表情差分的固定面部矩形：**x 782..928 / y 403..500**（1488×2272 画布）

## 待办（按优先级）

1. **人工筛选**：打开 `out/qa/gen15/*.png`，从 97 张里挑 25~40 张
   → 删掉 `out/gen15/picks.txt` 里不要的行
   → `python scripts/09_curate.py --picks out/gen15/picks.txt` 生成 `out/dataset/final/`
2. 补跑 `nugi02 / nugi04`（会发明长裙）：
   `--base-filter "nugi02,nugi04" --force-outfit "black lace lingerie, open white shirt, black pantyhose"`
3. 阶段 2：装 kohya（`SETUP.md` 最后一节）→ 训 SDXL 角色 LoRA
4. 若身份还原不够：加“脸部移植”局部重绘（denoise 0.35，与阶段 4 同一机制）

## 数据集构成提醒

- `out/dataset/char/` = 14 张单人立绘（身份锚点，训练时占比建议 ≥50%）
- `out/dataset/char_plus/` = 32 张（含双人图、事件 CG、16 张场景图，`_review.csv` 里标了 `NSFW_SCENE`）
- `out/gen15/` 的图是“底模脸”，用于补姿势；身份细节靠上面 14 张兜底
