# 素材分类报告（新数据集）

- 扫描：151 个文件

- `empty`：1
- `event_cg`：1
- `face_patch`：54
- `layer`：8
- `pose_cg`：16
- `sprite`：16
- `sprite_multi`：1
- `storyboard`：54

## 训练集

- `out/dataset/char/`：14 张（纯单人立绘，身份训练主力）
- `out/dataset/char_plus/`：32 张（+ 双人/场景/事件CG，补姿势）
- `out/dataset/faces/`：18 张唯一表情差分

## 差分画布规格

- `1488x2272`：11 张，组：hukumomi, hukusimi, lob, normal
  - 面部区并集：x=792..913, y=405..500
- `868x875`：6 张，组：go_f01, go_f02, go_f03, go_f04, go_f05, go_f06
  - 面部区并集：x=345..655, y=108..301
- `1200x800`：1 张，组：masage_odoroki
  - 面部区并集：x=927..1090, y=72..193

## 重复文件（哈希相同）

- 常装1.png == lob.png
- 常装3.png == nugi01.png
- hukumomi_1.png == hukusimi_1.png == lob_1.png == lob_default.png == normal_1.png == normal_default.png == nugi04_1.png == nugi04_default.png
- hukumomi_2.png == hukusimi_2.png == lob_2.png == normal_2.png == nugi04_2.png
- hukumomi_7.png == lob_7.png == normal_7.png == nugi04_7.png
- hukusimi_3.png == lob_3.png == normal_3 - コピー.png == normal_3.png == nugi04_3.png
- hukusimi_5.png == lob_5.png == normal_5 - コピー.png == normal_5.png == nugi04_5.png
- lob_10.png == normal_10.png == nugi04_10.png
- lob_4.png == normal_4 - コピー.png == normal_4.png == nugi04_4 - コピー.png == nugi04_4.png
- lob_6.png == normal_6.png == nugi04_6 - コピー.png == nugi04_6.png
- lob_8.png == normal_8.png == nugi04_8.png
- lob_9.png == normal_9.png == nugi04_9.png
- normal_11.png == nugi04_11.png