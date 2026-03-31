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
  const explainEl = document.getElementById("default-analysis-explain");
  explainEl.innerHTML = `
    <h3 style="margin:0 0 8px;">1) 数据如何转成可分析样本</h3>
    <ul>
      <li>输入来自 EEG 特征文件（trial级，常用 key 如 \`de_LDS_1..80\`）。</li>
      <li>每个 trial 的三维特征先在时间窗口上做均值聚合，得到固定长度向量。</li>
      <li>真实标签来自 \`save_info\`：从电影路径中提取情绪类别（如 happy/sad/anger）。</li>
    </ul>

    <h3 style="margin:12px 0 8px;">2) 如何做情绪分析</h3>
    <ul>
      <li>在同一批 trial 特征上训练/评估 LR、MLP 等模型，输出情绪类别预测。</li>
      <li><strong>subject_dependent</strong>：同一被试内划分训练/测试，关注个体内识别能力。</li>
      <li><strong>subject_independent</strong>：按被试分组做跨被试评估，关注泛化能力。</li>
    </ul>

    <h3 style="margin:12px 0 8px;">3) 指标怎么读</h3>
    <ul>
      <li><strong>Accuracy</strong>：整体正确率，越高越好。</li>
      <li><strong>Macro-F1</strong>：各情绪类别 F1 的平均，更能反映类别不均衡时的表现。</li>
      <li>报告中的 \`mean ± std\` 表示多折/多被试评估下的平均水平与波动。</li>
    </ul>
  `;

  // 默认：按 accuracy_mean 排序，报告页用于“看结果”
  renderTable(data.summary, "table-subject-dependent", "subject_dependent", "accuracy_mean");
  renderTable(data.summary, "table-subject-independent", "subject_independent", "accuracy_mean");

  const fileInput = document.getElementById("upload-zip-report");
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

  function renderProtocolOverallRows(p) {
    const row = (name, m) => `
      <tr>
        <td>${name}</td>
        <td>${fmt(m.overall.accuracy)}</td>
        <td>${fmt(m.overall.macro_f1)}</td>
        <td>${m.coverage.valid_trials}</td>
        <td>${m.coverage.expected_trials}</td>
        <td>${(m.coverage.ratio * 100).toFixed(1)}%</td>
      </tr>
    `;
    return row("LR", p.lr) + row("MLP", p.mlp);
  }

  function renderPerSubjectRows(items) {
    if (!items || !items.length) {
      return `<tr><td colspan="4">无可用被试结果</td></tr>`;
    }
    return items
      .map(
        (x) => `<tr><td>${x.subject}</td><td>${fmt(x.accuracy)}</td><td>${fmt(x.macro_f1)}</td><td>${x.valid_trials}</td></tr>`
      )
      .join("");
  }

  function renderPerFoldRows(items) {
    if (!items || !items.length) {
      return `<tr><td colspan="5">无可用fold结果</td></tr>`;
    }
    return items
      .map(
        (x) =>
          `<tr><td>${x.fold}</td><td>${x.test_subjects.join(", ")}</td><td>${fmt(x.accuracy)}</td><td>${fmt(x.macro_f1)}</td><td>${x.valid_trials}</td></tr>`
      )
      .join("");
  }

  function renderRealtimeReport(result, fileInfo) {
    const sd = result.protocols.subject_dependent;
    const si = result.protocols.subject_independent;
    const distItems = Object.entries(result.counts).sort((a, b) => b[1] - a[1]);
    const distHtml = distItems
      .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td><td>${((v / Math.max(result.total_trials, 1)) * 100).toFixed(1)}%</td></tr>`)
      .join("");
    const cs = result.confidence_stats || {};

    reportEl.innerHTML = `
      <div class="card card-soft">
        <h3>实时上传报告（单文件）</h3>

        <div class="card card-soft" style="padding: 12px; margin-top: 12px;">
          <h4 style="margin:0 0 6px;">实验输入信息（Input）</h4>
          <p><strong>上传文件：</strong>${fileInfo.name}</p>
          <p><strong>文件大小：</strong>${fileInfo.size}</p>
          <p><strong>上传被试：</strong>${(result.uploaded_subjects || []).join(", ") || "无"}</p>
          <p><strong>特征类型：</strong>${result.feature_type_used}</p>
          <p><strong>模型类型：</strong>LR + MLP（SEED 两协议）</p>
        </div>

        <div style="height: 10px;"></div>

        <div class="card card-soft" style="padding: 12px; margin-top: 12px;">
          <h4 style="margin:0 0 6px;">结果概览（Prediction Summary）</h4>
          <p>共分析 <strong>${result.total_trials}</strong> 个有效 trial，主导情绪为 <strong>${result.dominant_label}</strong>。</p>
          <div class="kpi-grid">
            <div class="kpi"><div class="kpi-label">置信度均值</div><div class="kpi-value">${fmt(cs.mean ?? 0)}</div></div>
            <div class="kpi"><div class="kpi-label">置信度中位数</div><div class="kpi-value">${fmt(cs.median ?? 0)}</div></div>
            <div class="kpi"><div class="kpi-label">置信度最小值</div><div class="kpi-value">${fmt(cs.min ?? 0)}</div></div>
            <div class="kpi"><div class="kpi-label">置信度最大值</div><div class="kpi-value">${fmt(cs.max ?? 0)}</div></div>
          </div>
        </div>

        <div style="height: 10px;"></div>

        <div class="card card-soft" style="padding: 12px; margin-top: 12px;">
          <h4 style="margin:0 0 6px;">SEED 协议指标（Accuracy / Macro-F1）</h4>
          <h5 style="margin:8px 0;">subject_dependent</h5>
          <table>
            <thead><tr><th>模型</th><th>Accuracy</th><th>Macro-F1</th><th>有效trial</th><th>期望trial</th><th>Coverage</th></tr></thead>
            <tbody>${renderProtocolOverallRows(sd)}</tbody>
          </table>
          <h5 style="margin:8px 0;">subject_independent</h5>
          <table>
            <thead><tr><th>模型</th><th>Accuracy</th><th>Macro-F1</th><th>有效trial</th><th>期望trial</th><th>Coverage</th></tr></thead>
            <tbody>${renderProtocolOverallRows(si)}</tbody>
          </table>
          <p class="subtitle">说明：指标基于有效覆盖样本计算（valid/expected）。</p>
        </div>

        <div class="card card-soft" style="padding: 12px; margin-top: 12px;">
          <h4 style="margin:0 0 6px;">subject_dependent 按被试指标</h4>
          <table>
            <thead><tr><th>subject</th><th>Accuracy</th><th>Macro-F1</th><th>有效trial</th></tr></thead>
            <tbody>${renderPerSubjectRows(sd.lr.per_subject)}</tbody>
          </table>
        </div>

        <div class="card card-soft" style="padding: 12px; margin-top: 12px;">
          <h4 style="margin:0 0 6px;">subject_independent 按fold指标（LR）</h4>
          <table>
            <thead><tr><th>fold</th><th>test_subjects</th><th>Accuracy</th><th>Macro-F1</th><th>有效trial</th></tr></thead>
            <tbody>${renderPerFoldRows(si.lr.per_fold)}</tbody>
          </table>
        </div>

        <div class="card card-soft" style="padding: 12px; margin-top: 12px;">
          <h4 style="margin:0 0 6px;">预测分布（Counts / Share）</h4>
          <table>
            <thead><tr><th>情绪</th><th>预测次数</th><th>占比</th></tr></thead>
            <tbody>${distHtml}</tbody>
          </table>
        </div>

        <p class="subtitle">实时实验报告仅展示结果与指标，不附带默认解析说明。</p>
      </div>
    `;
  }

  runBtn.addEventListener("click", async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      statusEl.textContent = "请先选择 .zip 文件。";
      return;
    }
    statusEl.textContent = "正在解析 zip 并生成协议报告，请稍候...";
    reportEl.innerHTML = "";
    runBtn.disabled = true;
    try {
      const result = await window.UploadInference.runUploadedZipAnalysis(file, featureSelect.value);
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

