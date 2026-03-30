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

  function renderRealtimeReport(result) {
    const distItems = Object.entries(result.counts).sort((a, b) => b[1] - a[1]);
    const distHtml = distItems.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

    reportEl.innerHTML = `
      <div class="card card-soft">
        <h3>实时上传报告（单文件）</h3>
        <p>共分析 <strong>${result.total_trials}</strong> 个 trial，主导情绪为 <strong>${result.dominant_label}</strong>。</p>
        <p>模型：<strong>${result.model_type}</strong>；特征：<strong>${result.feature_type_used}</strong>。</p>
        <table>
          <thead><tr><th>情绪</th><th>预测次数</th></tr></thead>
          <tbody>${distHtml}</tbody>
        </table>
        <p class="subtitle">
          解释：这是“上传文件推理报告”，反映该文件在当前模型下的情绪预测分布；与页面下方基线表格不同，后者是跨样本评估结果（Accuracy/Macro-F1）。
        </p>
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
      renderRealtimeReport(result);
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

