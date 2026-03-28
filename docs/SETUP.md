# 环境与数据重建说明

本仓库上传到 GitHub 时，已经主动排除了以下不适合版本管理的大文件或本地内容：

- `.venv/` 虚拟环境
- `data/` 下的原始视频、抽帧结果和高维特征
- `outputs/` 下的中间模型、分数和可视化产物
- `results/` 下可重复生成的导出结果

只要按本文档重新创建环境并补齐输入数据，就可以在新机器上恢复运行。

## 1. Python 版本

原项目环境使用的是 `Python 3.11.9`。建议优先使用同版本，避免 `torch`、`timm`、`transformers` 组合出现兼容性偏差。

## 2. 创建虚拟环境

### 方式 A：`venv`

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 方式 B：Conda

```bash
conda create -n money python=3.11.9
conda activate money
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. 初始化目录

仓库只保留了空目录占位，首次运行前建议手动创建下面这些目录：

```bash
mkdir -p data/videos
mkdir -p data/frames/train data/frames/test
mkdir -p data/features/timm
mkdir -p outputs/features outputs/features_2d
mkdir -p outputs/models outputs/models_2d
mkdir -p outputs/scores outputs/scores_2d
mkdir -p outputs/postprocess outputs/evaluation outputs/deliverables/PPT_Assets
mkdir -p results/final_for_ppt results/final_for_ppt_2d
```

## 4. 放置原始视频

把待处理道路监控视频放到 `data/videos/`，例如：

```text
data/videos/2.mp4
```

如果文件名不同，只需要在后面的命令里把 `--video` 参数替换成你的实际路径。

## 5. 抽帧

项目默认按 2 FPS 抽帧，并把前 12 分钟作为正常训练段、后续作为测试段：

```bash
python src/data_prep/extract_frames.py \
  --video data/videos/2.mp4 \
  --out-train data/frames/train \
  --out-test data/frames/test \
  --fps 2 \
  --train-minutes 12 \
  --size 224,224
```

## 6. 提取特征

### SigLIP

```bash
python src/feature_extraction/siglip_feature.py \
  --data data/frames/train \
  --out data/features/siglip_train.npy

python src/feature_extraction/siglip_feature.py \
  --data data/frames/test \
  --out data/features/siglip_test.npy
```

### TIMM 基线模型

下面这几个模型是当前项目里实际使用过的基线：

- `resnet101.a1h_in1k`
- `vit_base_patch16_224.augreg2_in21k_ft_in1k`
- `convnext_base.fb_in22k_ft_in1k`
- `efficientnet_b0.ra_in1k`

可以用下面这组命令批量提取：

```bash
for model in \
  resnet101.a1h_in1k \
  vit_base_patch16_224.augreg2_in21k_ft_in1k \
  convnext_base.fb_in22k_ft_in1k \
  efficientnet_b0.ra_in1k
do
  python src/feature_extraction/timm_feature.py \
    --data data/frames/train \
    --model "$model" \
    --out "data/features/timm/${model}_train.npy"

  python src/feature_extraction/timm_feature.py \
    --data data/frames/test \
    --model "$model" \
    --out "data/features/timm/${model}_test.npy"
done
```

## 7. 运行 2D 实验流水线

当前 `src/pipeline/pipeline_2d.py` 已改成自动识别仓库根目录，并使用你当前激活环境里的 Python：

```bash
python src/pipeline/pipeline_2d.py
```

运行后主要会生成：

- `outputs/features_2d/`
- `outputs/models_2d/`
- `outputs/scores_2d/`
- `results/final_for_ppt_2d/`

## 8. 可选：生成语义解释

如果已经生成高风险帧，可继续运行轻量级 BLIP 说明脚本：

```bash
python src/visualization/multimodal_explanation.py \
  --risk_dir results/final_for_ppt_2d/gaussian_risk_frames \
  --top_k 5
```

## 9. 额外说明

- 首次运行 `transformers` / `timm` 模型时，会联网下载预训练权重。
- 如果你不打算复现实验，只想查看项目结构和展示材料，只需阅读 `README.md`、`presentation/` 和 `提交文件/`。
- 如果后续要把数据或结果也纳入版本管理，先修改根目录 `.gitignore`，再有选择地提交。
