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

function renderTable(summary, protocol, metric) {
  const tbody = document.querySelector("#result-table tbody");
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

function renderConfusionSelector(data, protocol) {
  const cmSelect = document.getElementById("cm-select");
  const allKeys = [...Object.keys(data.ml_confusion || {}), ...Object.keys(data.dl_confusion || {})];
  const keys = allKeys.filter((k) => k.startsWith(protocol + "::"));
  cmSelect.innerHTML = keys.map((k) => `<option value="${k}">${k}</option>`).join("");
  if (keys.length > 0) {
    renderConfusionMatrix(data, keys[0]);
  } else {
    document.getElementById("cm-container").innerHTML = "<p>无可用混淆矩阵</p>";
  }
}

function renderConfusionMatrix(data, key) {
  const merged = { ...(data.ml_confusion || {}), ...(data.dl_confusion || {}) };
  const item = merged[key];
  if (!item) {
    document.getElementById("cm-container").innerHTML = "<p>未找到对应矩阵</p>";
    return;
  }
  const labels = item.labels;
  const matrix = item.matrix;
  let html = '<table class="matrix"><thead><tr><th>GT \\ Pred</th>';
  html += labels.map((l) => `<th>${l}</th>`).join("");
  html += "</tr></thead><tbody>";
  for (let i = 0; i < matrix.length; i++) {
    html += `<tr><th>${labels[i]}</th>`;
    html += matrix[i].map((v) => `<td>${fmt(v)}</td>`).join("");
    html += "</tr>";
  }
  html += "</tbody></table>";
  document.getElementById("cm-container").innerHTML = html;
}

async function main() {
  const resp = await fetch("./assets/results.json");
  const data = await resp.json();
  renderDatasetSummary(data.dataset);

  const protocolSelect = document.getElementById("protocol-select");
  const metricSelect = document.getElementById("metric-select");
  const cmSelect = document.getElementById("cm-select");

  function rerender() {
    renderTable(data.summary, protocolSelect.value, metricSelect.value);
    renderConfusionSelector(data, protocolSelect.value);
  }

  protocolSelect.addEventListener("change", rerender);
  metricSelect.addEventListener("change", () => renderTable(data.summary, protocolSelect.value, metricSelect.value));
  cmSelect.addEventListener("change", () => renderConfusionMatrix(data, cmSelect.value));

  // Upload inference
  const fileInput = document.getElementById("upload-mat");
  const featureSelect = document.getElementById("feature-type-upload");
  const runBtn = document.getElementById("run-upload-infer");
  const statusEl = document.getElementById("upload-status");
  const summaryEl = document.getElementById("upload-summary");
  const tableEl = document.getElementById("upload-table");

  function renderUploadResult(result) {
    const distHtml = Object.entries(result.counts)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `<span class="chip">${k}: ${v}</span>`)
      .join("");
    summaryEl.innerHTML = `
      <div class="card card-soft">
        <p><strong>分析完成</strong>：共 ${result.total_trials} 个 trial，主导情绪为 <strong>${result.dominant_label}</strong></p>
        <p><strong>模型</strong>：${result.model_type}，<strong>特征类型</strong>：${result.feature_type_used}</p>
        <p>${distHtml}</p>
      </div>
    `;
    const rows = result.rows
      .slice(0, 20)
      .map(
        (r) => `
          <tr>
            <td>${r.trial}</td>
            <td>${r.pred_label}</td>
            <td>${fmt(r.confidence)}</td>
          </tr>
        `
      )
      .join("");
    tableEl.innerHTML = `
      <table>
        <thead>
          <tr><th>trial</th><th>预测情绪</th><th>置信度</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="subtitle">仅展示前 20 条预测，完整结果可在后续版本增加导出。</p>
    `;
  }

  runBtn.addEventListener("click", async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      statusEl.textContent = "请先选择 .mat 文件。";
      return;
    }
    statusEl.textContent = "正在加载推理环境并解析 .mat，请稍候...";
    summaryEl.innerHTML = "";
    tableEl.innerHTML = "";
    runBtn.disabled = true;
    try {
      const result = await window.UploadInference.runUploadedMatInference(file, featureSelect.value);
      statusEl.textContent = "上传分析完成。";
      renderUploadResult(result);
    } catch (err) {
      statusEl.textContent = `分析失败：${String(err)}`;
    } finally {
      runBtn.disabled = false;
    }
  });

  rerender();
}

main().catch((err) => {
  document.body.innerHTML = `<pre>加载失败: ${String(err)}</pre>`;
});

