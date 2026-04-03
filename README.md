# SEED 情绪识别项目（ML优先 + DL进阶 + GitHub Pages）

本项目基于 `data/EEG_features` 的 `.mat` 特征文件，实现了一个从数据清单构建到模型训练、可视化和网页展示的完整流程，适合入门者边学边做。

## 1. 项目结构

- `src/data/build_manifest.py`：构建 trial 级 manifest 与标签映射
- `src/ml/train_ml.py`：机器学习基线（LR/SVM/RF）
- `src/dl/train_mlp.py`：深度学习基线（MLP）
- `src/ml/export_web_protocol_models.py`：导出网页端协议模型（LR+MLP；subject_dependent / subject_independent）
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
4. 导出网页端协议模型 JSON（用于 zip 上传后计算 Accuracy/Macro-F1）
5. 生成图表和网页数据 `docs/assets/results.json`

## 4. 单步运行

```bash
python src/data/build_manifest.py
python src/ml/train_ml.py
python src/dl/train_mlp.py
python src/ml/export_web_protocol_models.py
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
- 上传 zip 协议模型：`docs/assets/protocol_models_*.json`
- 自动部署工作流：`.github/workflows/deploy-pages.yml`

## 7. 样本集（交互 / 即时报告上传用 zip）

`Sample/` 目录下提供了若干可直接在 **`docs/demo.html`** 与 **`docs/report.html`** 中上传测试的压缩包。它们与网页端约定一致：zip 根目录下必须包含 **`EEF_features/`** 与 **`save_info/`** 两个子目录，前者放置与被试编号对应的 `.mat` 特征文件，后者放置由实验日期与 session 编号构成的 `*_save_info.csv`，以便将 trial 与情绪标签对齐。若将文件平铺在 zip 根目录或缺少上述子目录名，页面将无法识别被试与 trial。

以单被试包为例，目录结构如下（多被试时仅在两个目录中各增加对应文件）：

```text
EEF_features/
  1.mat
save_info/
  1_20221001_1_save_info.csv
  1_20221005_2_save_info.csv
  1_20221006_3_save_info.csv
  1_20221008_4_save_info.csv
```

当前预置文件说明：

| 文件名 | 内容 |
|--------|------|
| `sample_upload_subject1.zip` | 被试 1，四个 session 的 `save_info` + 对应 `1.mat` |
| `sample_upload_subject2.zip` | 被试 2，单被试完整包 |
| `sample_upload_subject10.zip` | 被试 10，单被试完整包 |
| `sample_upload_subjects_2_10.zip` | 被试 2 与 10，用于测试**多被试**同包上传 |

在本地已具备 `data/EEG_features/*.mat` 与 `data/save_info` 时，可用脚本重新生成上述 zip（输出目录默认为 `Sample/`）：

```bash
python scripts/build_sample_upload_zips.py
```

可选参数：`--eeg-features-dir`、`--save-info-dir`、`--out-dir`。