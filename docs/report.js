function fmt(value) {
  return Number(value).toFixed(4);
}

function renderDatasetSummary(dataset) {
  const el = document.getElementById("dataset-summary");
  const labels = Object.entries(dataset.labels)
    .map(([k, v]) => `<span class="chip">${k}: ${v}</span>`)
    .join("");
  el.innerHTML = `
    <div><strong>样本数：</strong>${dataset.samples}</div>
    <div><strong>被试数：</strong>${dataset.subjects}</div>
    <div><strong>每被试 trial：</strong>${dataset.per_subject_trials.join(", ")}</div>
    <div><strong>特征类型：</strong>${dataset.feature_type}</div>
    <div style="grid-column:1 / -1;"><strong>标签分布：</strong><br/>${labels}</div>
  `;
}

function renderTable(summary, tableId, protocol, metric) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  const rows = summary.filter((r) => r.protocol === protocol).sort((a, b) => b[metric] - a[metric]);
  tbody.innerHTML = rows
    .map(
      (r) => `
      <tr>
        <td>${r.model}</td>
        <td>${fmt(r.accuracy_mean)} ± ${fmt(r.accuracy_std)}</td>
        <td>${fmt(r.macro_f1_mean)} ± ${fmt(r.macro_f1_std)}</td>
        <td>${r.num_folds}</td>
      </tr>
    `
    )
    .join("");
}

async function main() {
  const resp = await fetch("./assets/results.json");
  const data = await resp.json();

  renderDatasetSummary(data.dataset);

  // 默认：按 accuracy_mean 排序，报告页用于“看结果”
  renderTable(data.summary, "table-subject-dependent", "subject_dependent", "accuracy_mean");
  renderTable(data.summary, "table-subject-independent", "subject_independent", "accuracy_mean");

  const fileInput = document.getElementById("upload-mat-report");
  const featureSelect = document.getElementById("feature-type-report");
  const runBtn = document.getElementById("run-report-infer");
  const statusEl = document.getElementById("report-upload-status");
  const reportEl = document.getElementById("realtime-report");

  function humanBytes(bytes) {
    const units = ["B", "KB", "MB", "GB"];
    let v = bytes;
    let idx = 0;
    while (v >= 1024 && idx < units.length - 1) {
      v /= 1024;
      idx += 1;
    }
    return `${v.toFixed(2)} ${units[idx]}`;
  }

  function renderRealtimeReport(result, fileInfo) {
    const distItems = Object.entries(result.counts).sort((a, b) => b[1] - a[1]);
    const cs = result.confidence_stats || {};
    const meanPct = (v) => ((v / Math.max(result.total_trials, 1)) * 100).toFixed(1);
    const distHtmlPct = distItems
      .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td><td>${meanPct(v)}%</td></tr>`)
      .join("");

    const topTrials = result.top_trials_by_confidence || [];
    const topTrialsHtml = topTrials
      .map((r) => `<tr><td>${r.trial}</td><td>${r.pred_label}</td><td>${fmt(r.confidence)}</td></tr>`)
      .join("");

    const perLabelMeanConf = result.per_label_confidence_mean || {};
    const perLabelHtml = Object.entries(perLabelMeanConf)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v)}</td></tr>`)
      .join("");

    reportEl.innerHTML = `
      <div class="card card-soft">
        <h3>实时上传报告（单文件）</h3>

        <div class="card card-soft" style="padding: 12px; margin-top: 12px;">
          <h4 style="margin:0 0 6px;">实验输入信息（Input）</h4>
          <p><strong>上传文件：</strong>${fileInfo.name}</p>
          <p><strong>文件大小：</strong>${fileInfo.size}</p>
          <p><strong>特征类型：</strong>${result.feature_type_used}</p>
          <p><strong>模型类型：</strong>${result.model_type}</p>
        </div>

        <div style="height: 10px;"></div>

        <div class="card card-soft" style="padding: 12px; margin-top: 12px;">
          <h4 style="margin:0 0 6px;">结果概览（Prediction Summary）</h4>
          <p>共分析 <strong>${result.total_trials}</strong> 个 trial，主导情绪为 <strong>${result.dominant_label}</strong>。</p>
          <div class="kpi-grid">
            <div class="kpi"><div class="kpi-label">置信度均值</div><div class="kpi-value">${fmt(cs.mean ?? 0)}</div></div>
            <div class="kpi"><div class="kpi-label">置信度中位数</div><div class="kpi-value">${fmt(cs.median ?? 0)}</div></div>
            <div class="kpi"><div class="kpi-label">置信度最小值</div><div class="kpi-value">${fmt(cs.min ?? 0)}</div></div>
            <div class="kpi"><div class="kpi-label">置信度最大值</div><div class="kpi-value">${fmt(cs.max ?? 0)}</div></div>
          </div>
        </div>

        <div style="height: 10px;"></div>

        <div class="card card-soft" style="padding: 12px; margin-top: 12px;">
          <h4 style="margin:0 0 6px;">预测分布（Counts / Share）</h4>
          <table>
            <thead><tr><th>情绪</th><th>预测次数</th><th>占比</th></tr></thead>
            <tbody>${distHtmlPct}</tbody>
          </table>
          <p class="subtitle" style="margin-top: 10px;">
            解释：这是“上传文件推理报告”，反映你上传的特征文件在当前模型下的情绪预测分布。
            由于上传 .mat 本身不携带“真实标签”，本页面不直接给出 Accuracy / Macro-F1 等评估指标。
          </p>
        </div>

        <div class="card card-soft" style="padding: 12px; margin-top: 12px;">
          <h4 style="margin:0 0 6px;">置信度（Confidence）</h4>
          <p class="subtitle" style="margin-top: 0;">
            说明：置信度来自当前模型输出的 softmax 概率（取被预测类别对应的概率）。
          </p>
          <table>
            <thead><tr><th>情绪</th><th>作为预测结果时的平均置信度</th></tr></thead>
            <tbody>${perLabelHtml}</tbody>
          </table>
        </div>

        <div class="card card-soft" style="padding: 12px; margin-top: 12px;">
          <h4 style="margin:0 0 6px;">最高置信度 trial（Top ${topTrials.length}）</h4>
          <table>
            <thead><tr><th>trial</th><th>预测情绪</th><th>置信度</th></tr></thead>
            <tbody>${topTrialsHtml}</tbody>
          </table>
          <p class="subtitle">说明：只展示最高置信度的前 ${topTrials.length} 条，便于快速检查。</p>
        </div>
      </div>
    `;
  }

  runBtn.addEventListener("click", async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      statusEl.textContent = "请先选择 .mat 文件。";
      return;
    }
    statusEl.textContent = "正在生成实时报告，请稍候...";
    reportEl.innerHTML = "";
    runBtn.disabled = true;
    try {
      const result = await window.UploadInference.runUploadedMatInference(file, featureSelect.value);
      statusEl.textContent = "实时报告生成完成。";
      const fileInfo = { name: file.name, size: humanBytes(file.size || 0) };
      renderRealtimeReport(result, fileInfo);
    } catch (err) {
      statusEl.textContent = `报告生成失败：${String(err)}`;
    } finally {
      runBtn.disabled = false;
    }
  });
}

main().catch((err) => {
  document.body.innerHTML = `<pre>加载失败: ${String(err)}</pre>`;
});

