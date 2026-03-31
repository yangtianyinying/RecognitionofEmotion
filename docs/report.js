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

function findSummary(summary, protocol, model) {
  return summary.find((x) => x.protocol === protocol && x.model === model);
}

async function main() {
  const resp = await fetch("./assets/results.json");
  const data = await resp.json();

  renderDatasetSummary(data.dataset);
  const explainEl = document.getElementById("default-analysis-explain");
  const sdLR = findSummary(data.summary, "subject_dependent", "logistic_regression");
  const sdRF = findSummary(data.summary, "subject_dependent", "random_forest");
  const sdSVM = findSummary(data.summary, "subject_dependent", "svm_rbf");
  const sdMLP = findSummary(data.summary, "subject_dependent", "mlp");
  const siLR = findSummary(data.summary, "subject_independent", "logistic_regression");
  const siRF = findSummary(data.summary, "subject_independent", "random_forest");
  const siSVM = findSummary(data.summary, "subject_independent", "svm_rbf");
  const siMLP = findSummary(data.summary, "subject_independent", "mlp");

  explainEl.innerHTML = `
    <h2 style="margin:0 0 10px;">报告解析</h2>

    <h3 style="margin:10px 0 8px;">数据转换</h3>
    <p>
      默认报告基于 trial 级 EEG 特征进行构建。系统先读取 \`de_LDS_1..80\`（或同结构的 \`de_*\`、\`psd_*\`）特征键，再对每个 trial 的时间窗口特征做均值聚合，
      得到固定长度向量。与此同时，真实标签由 \`save_info\` 提供：通过电影路径中的情绪目录名提取类别，最终形成“每个 trial 一条特征向量 + 一个情绪标签”的监督样本。
      这个转换过程的核心价值在于，把原始复杂时序信号转化为可比较、可统计、可复现实验的结构化数据。
    </p>

    <h3 style="margin:12px 0 8px;">情绪分析</h3>
    <p>
      在默认报告中，模型比较分为传统机器学习（Logistic Regression、Random Forest、SVM-RBF）与深度学习基线（MLP）两组。结果显示，在
      <code>subject_dependent</code>
      场景下，Logistic Regression 的 Accuracy 约为 <strong>${fmt(sdLR?.accuracy_mean ?? 0)}</strong>、Macro-F1 约为
      <strong>${fmt(sdLR?.macro_f1_mean ?? 0)}</strong>，明显高于 Random Forest（Accuracy
      <strong>${fmt(sdRF?.accuracy_mean ?? 0)}</strong>）和 MLP（Accuracy <strong>${fmt(sdMLP?.accuracy_mean ?? 0)}</strong>）。
      这说明在被试内设置里，线性基线对当前特征表达具有较强适配性。到了 <code>subject_independent</code> 场景，MLP 的 Accuracy 约为
      <strong>${fmt(siMLP?.accuracy_mean ?? 0)}</strong>，与 LR 的 <strong>${fmt(siLR?.accuracy_mean ?? 0)}</strong> 接近，
      但 Macro-F1 上 LR（<strong>${fmt(siLR?.macro_f1_mean ?? 0)}</strong>）略高于 MLP（<strong>${fmt(siMLP?.macro_f1_mean ?? 0)}</strong>），
      反映出跨被试泛化时，深度模型和线性模型各有取舍。
    </p>

    <h3 style="margin:12px 0 8px;">指标解读</h3>
    <p>
      Accuracy 用于回答“总体上预测对了多少”，而 Macro-F1 用于回答“各个情绪类别是否都被公平地识别”。当一个模型 Accuracy 不低但 Macro-F1 明显偏低时，
      常见原因是模型对多数类别更友好、对少数类别识别不足。报告中的
      <code>mean ± std</code>
      表示多折或多被试评估下的平均性能与波动范围：均值越高说明整体能力越强，标准差越小说明稳定性越好。
    </p>

    <h3 style="margin:12px 0 8px;">数据分析总结</h3>
    <p>
      结合表格与图像可以得到一个较清晰的结论：当前特征体系下，默认基线在被试内场景明显优于跨被试场景，说明“个体差异”仍是核心挑战。图1与图2显示不同模型在两协议下存在一致的性能下滑趋势，
      图3（混淆矩阵）中的非对角元素也提示了情绪间可分性仍有限，尤其在语义接近或唤醒度相近的类别上更容易互相误判。整体而言，默认报告证明了流程可复现、模型可比较，但若要进一步提升跨被试能力，
      仍需在特征鲁棒性、域间对齐与模型泛化策略上继续迭代。
    </p>
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

