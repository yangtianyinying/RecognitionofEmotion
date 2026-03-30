# GitHub Pages 使用说明（SEED 情绪识别）

本目录下提供三个页面：

1. `index.html`：说明页（你应该先打开这个，了解怎么用网站）
2. `report.html`：报告页（主要看汇总表格与静态图表）
3. `demo.html`：交互页（下拉框切换协议/指标/混淆矩阵，查看不同模型结果）

所有页面的数据都来自：`docs/assets/results.json`（由本地训练脚本自动生成）。

## 1) 不上传文件：看“默认基线报告”

打开 `report.html`：

- 页面会展示两种评估协议的汇总结果表（`subject_dependent` / `subject_independent`）
- 同时展示训练导出的静态图表（accuracy / macro-f1 / 混淆矩阵平均图）

这个模式适合直接写作业或论文的结果章节。

## 2) 上传文件：生成“实时上传子集协议报告”

在 `demo.html` 或 `report.html` 上传 `.zip` 文件，结构固定如下：

```text
your_upload.zip
├─ EEF_features/
│  ├─ 1.mat
│  ├─ 5.mat
│  └─ 12.mat
└─ save_info/
   ├─ 1_20221001_1_save_info.csv
   ├─ 1_20221001_2_save_info.csv
   ├─ 1_20221001_3_save_info.csv
   ├─ 1_20221001_4_save_info.csv
   ├─ 5_20221020_1_save_info.csv
   └─ ...
```

必须满足：

1. `EEF_features/<subject>.mat`
2. `save_info/<subject>_<date>_<session>_save_info.csv`
3. session 建议覆盖 1..4（便于 trial 1..80 对齐更完整）
4. 选择“特征类型”（`de_lds` / `de` / `psd`）后点击分析

页面的输出：

- 预测分布：该文件中所有 trial 的预测情绪次数与占比
- 置信度统计：置信度均值/中位数/最小/最大（置信度来自模型 softmax 概率）
- 最高置信度 trial：展示前 15 条预测，便于快速观察
- 协议指标：`subject_dependent` 与 `subject_independent` 的 Accuracy / Macro-F1
- 覆盖率：显示有效 trial / 期望 trial（缺失 session 时会下降）

重要说明：

- 网页端推理使用导出的协议模型（LR + MLP）：
  - `docs/assets/protocol_models_lr_subject.json`
  - `docs/assets/protocol_models_mlp_subject.json`
  - `docs/assets/protocol_models_lr_fold.json`
  - `docs/assets/protocol_models_mlp_fold.json`
- 报告中的 Accuracy / Macro-F1 基于“有效覆盖样本”计算（若某些 session 缺失，会看到 valid/expected 下降）。

首次加载时的提示：

- 浏览器需要加载 Pyodide（解析 `.mat`）与 JSZip（解压 zip），首次点击可能稍慢，这是正常现象。
