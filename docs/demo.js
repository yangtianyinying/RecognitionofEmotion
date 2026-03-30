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

  // Upload zip inference
  const fileInput = document.getElementById("upload-zip");
  const featureSelect = document.getElementById("feature-type-upload");
  const runBtn = document.getElementById("run-upload-infer");
  const statusEl = document.getElementById("upload-status");
  const summaryEl = document.getElementById("upload-summary");
  const tableEl = document.getElementById("upload-table");

  function renderProtocolMetricRow(name, m) {
    if (!m || !m.overall) {
      return `<tr><td>${name}</td><td>-</td><td>-</td><td>0</td><td>0</td></tr>`;
    }
    return `
      <tr>
        <td>${name}</td>
        <td>${fmt(m.overall.accuracy)}</td>
        <td>${fmt(m.overall.macro_f1)}</td>
        <td>${m.coverage.valid_trials}</td>
        <td>${m.coverage.expected_trials}</td>
      </tr>
    `;
  }

  function renderUploadResult(result) {
    const sd = result.protocols.subject_dependent;
    const si = result.protocols.subject_independent;
    const distItems = Object.entries(result.counts).sort((a, b) => b[1] - a[1]);
    const distRowsHtml = distItems
      .map(([k, v]) => {
        const pct = (v / Math.max(result.total_trials, 1)) * 100;
        return `<tr><td>${k}</td><td>${v}</td><td>${pct.toFixed(1)}%</td></tr>`;
      })
      .join("");

    const topRows = (result.top_trials_by_confidence || []).map(
      (r) => `
        <tr>
          <td>${r.subject}</td>
          <td>${r.trial}</td>
          <td>${r.pred_label}</td>
          <td>${fmt(r.confidence)}</td>
        </tr>
      `
    );

    const cs = result.confidence_stats || {};
    const uploadedSubjects = (result.uploaded_subjects || []).join(", ");
    summaryEl.innerHTML = `
      <div class="card card-soft">
        <p><strong>上传分析完成</strong>：共 ${result.total_trials} 个有效 trial，主导情绪为 <strong>${result.dominant_label}</strong></p>
        <p><strong>上传被试数</strong>：${result.uploaded_subjects_count}（subject: ${uploadedSubjects || "无"}）</p>
        <p><strong>特征</strong>：${result.feature_type_used}；<strong>模型</strong>：LR + MLP（SEED 两协议）</p>
        <div class="kpi-grid">
          <div class="kpi"><div class="kpi-label">置信度均值</div><div class="kpi-value">${fmt(cs.mean ?? 0)}</div></div>
          <div class="kpi"><div class="kpi-label">置信度中位数</div><div class="kpi-value">${fmt(cs.median ?? 0)}</div></div>
          <div class="kpi"><div class="kpi-label">置信度最小值</div><div class="kpi-value">${fmt(cs.min ?? 0)}</div></div>
          <div class="kpi"><div class="kpi-label">置信度最大值</div><div class="kpi-value">${fmt(cs.max ?? 0)}</div></div>
        </div>
      </div>
      <div style="height: 10px;"></div>
      <div class="card card-soft">
        <h3 style="margin:0 0 10px;">上传子集协议指标（Accuracy / Macro-F1）</h3>
        <h4 style="margin:10px 0 8px;">subject_dependent</h4>
        <table>
          <thead><tr><th>模型</th><th>Accuracy</th><th>Macro-F1</th><th>有效trial</th><th>期望trial</th></tr></thead>
          <tbody>
            ${renderProtocolMetricRow("LR", sd.lr)}
            ${renderProtocolMetricRow("MLP", sd.mlp)}
          </tbody>
        </table>
        <h4 style="margin:10px 0 8px;">subject_independent</h4>
        <table>
          <thead><tr><th>模型</th><th>Accuracy</th><th>Macro-F1</th><th>有效trial</th><th>期望trial</th></tr></thead>
          <tbody>
            ${renderProtocolMetricRow("LR", si.lr)}
            ${renderProtocolMetricRow("MLP", si.mlp)}
          </tbody>
        </table>
        <p class="subtitle">
          说明：若 zip 中某些 session 缺失或无法对齐到真实标签，指标会按有效覆盖样本计算（valid/expected）。
        </p>
      </div>
    `;

    tableEl.innerHTML = `
      <div class="card card-soft">
        <h3 style="margin:0 0 10px;">预测分布（次数 / 占比）</h3>
        <table>
          <thead><tr><th>情绪</th><th>次数</th><th>占比</th></tr></thead>
          <tbody>${distRowsHtml}</tbody>
        </table>
      </div>
      <div style="height: 10px;"></div>
      <div class="card card-soft">
        <h3 style="margin:0 0 10px;">最高置信度的 15 条 trial（跨上传子集）</h3>
        <table>
          <thead><tr><th>subject</th><th>trial</th><th>预测情绪</th><th>置信度</th></tr></thead>
          <tbody>${topRows.join("")}</tbody>
        </table>
        <p class="subtitle">说明：置信度来自模型输出 softmax 概率（参考 subject_independent LR 预测集合）。</p>
      </div>
    `;
  }

  runBtn.addEventListener("click", async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      statusEl.textContent = "请先选择 .zip 文件。";
      return;
    }
    statusEl.textContent = "正在加载环境并解析 zip（mat + csv），请稍候...";
    summaryEl.innerHTML = "";
    tableEl.innerHTML = "";
    runBtn.disabled = true;
    try {
      const result = await window.UploadInference.runUploadedZipAnalysis(file, featureSelect.value);
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

