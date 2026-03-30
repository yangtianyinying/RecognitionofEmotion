# SEED 情绪识别项目（ML优先 + DL进阶 + GitHub Pages）

本项目基于 `data/EEG_features` 的 `.mat` 特征文件，实现了一个从数据清单构建到模型训练、可视化和网页展示的完整流程，适合入门者边学边做。

## 1. 项目结构

- `src/data/build_manifest.py`：构建 trial 级 manifest 与标签映射
- `src/ml/train_ml.py`：机器学习基线（LR/SVM/RF）
- `src/dl/train_mlp.py`：深度学习基线（MLP）
- `src/visualization/build_assets.py`：汇总结果并生成图表与网页 JSON
- `outputs/json/`：实验结果（可复用）
- `outputs/figures/`：静态图表
- `docs/`：GitHub Pages 站点源码

## 2. 环境安装

```bash
python -m pip install -r requirements.txt
```

## 3. 一键运行

```bash
python run_pipeline.py
```

会依次执行：

1. 生成 `manifest.jsonl`
2. 训练 ML 基线
3. 训练 DL 基线（MLP）
4. 生成图表和网页数据 `docs/assets/results.json`

## 4. 单步运行

```bash
python src/data/build_manifest.py
python src/ml/train_ml.py
python src/dl/train_mlp.py
python src/visualization/build_assets.py
```

## 5. 当前实验协议

- 数据单位：trial-level（将每个 trial 的窗口特征在时间维做均值聚合）
- 特征：`de_LDS_*`（可切换 `de` / `psd`）
- 协议：
  - `subject_dependent`：被试内 8:2 划分
  - `subject_independent`：按被试分组 5 折
- 指标：Accuracy、Macro-F1、平均混淆矩阵

## 6. GitHub Pages

- 页面源码：`docs/`
- 数据来源：`docs/assets/results.json`
- 自动部署工作流：`.github/workflows/deploy-pages.yml`

# RecognitionofEmotion
# RecognitionofEmotion
