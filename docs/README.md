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

## 2) 上传文件：生成“实时上传报告 / 预测分布”

在 `demo.html` 或 `report.html` 上传 `.mat` 文件：

1. 上传的文件必须是本项目特征导出的 `.mat`（也就是与你本地 `data/EEG_features` 同结构的文件）
2. 选择“特征类型”（`de_lds` / `de` / `psd`），让网页能在 `.mat` 内部找到对应 key
3. 点击按钮开始分析/生成报告

页面的输出：

- 预测分布：该文件中所有 trial 的预测情绪次数与占比
- 置信度统计：置信度均值/中位数/最小/最大（置信度来自模型 softmax 概率）
- 最高置信度 trial：展示前 15 条预测，便于快速观察

重要说明：

- 网页端推理目前只使用导出的 `Logistic Regression` 模型（`docs/assets/web_model.json`）
- 由于上传 `.mat` 通常不携带真实标签，**实时上传报告不会直接给 Accuracy / Macro-F1**

首次加载时的提示：

- 浏览器需要加载 Pyodide 来解析 `.mat`，首次点击可能稍慢，这是正常现象。
