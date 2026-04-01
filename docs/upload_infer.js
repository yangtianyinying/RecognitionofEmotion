let pyodideReady = null;
/** 并发安全：多入口同时 await 时只初始化一次 Pyodide */
let pyodideInitPromise = null;
let protocolModelsCache = null;
let globalWebModelCache = null;

const PYODIDE_VERSION = "0.27.2";

/** 多个 indexURL，用于 jsDelivr 不可达时回退（如部分网络环境）。须与 scipy 等包同源。 */
const PYODIDE_INDEX_URLS = [
  `https://unpkg.com/pyodide@${PYODIDE_VERSION}/`,
  `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`,
  `https://cdn.jsdelivr.net/npm/pyodide@${PYODIDE_VERSION}/`,
  `https://fastly.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`,
];

function normalizePyodideIndexURL(base) {
  const s = String(base).trim();
  return s.endsWith("/") ? s : `${s}/`;
}

function ensureScript(src) {
  return new Promise((resolve, reject) => {
    const existing = [...document.getElementsByTagName("script")].find((s) => s.src === src);
    if (existing) {
      resolve();
      return;
    }
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`脚本加载失败: ${src}`));
    document.head.appendChild(s);
  });
}

function removePyodideScript(scriptUrl) {
  const prev = [...document.getElementsByTagName("script")].find((x) => x.src === scriptUrl);
  if (prev) prev.remove();
  try {
    delete window.loadPyodide;
  } catch {
    /* ignore */
  }
}

async function initPyodideOnce(indexURL) {
  const base = normalizePyodideIndexURL(indexURL);
  const scriptUrl = `${base}pyodide.js`;
  await ensureScript(scriptUrl);
  if (typeof window.loadPyodide !== "function") {
    removePyodideScript(scriptUrl);
    throw new Error("loadPyodide 未定义");
  }
  const py = await window.loadPyodide({ indexURL: base });
  // 须分别加载：解析 .mat 用 numpy + scipy.io；部分环境下数组参数可能未完整安装 numpy
  await py.loadPackage("numpy");
  await py.loadPackage("scipy");
  await py.runPythonAsync("import numpy\nimport scipy.io");
  return py;
}

/**
 * 在运行含 numpy 的代码前调用。幂等；可修复「旧缓存 JS 只 load 了 scipy」导致的 numpy 缺失。
 */
async function ensurePyodideNumericLibs(py) {
  await py.loadPackage("numpy");
  await py.loadPackage("scipy");
}

async function getPyodide() {
  if (pyodideReady) return pyodideReady;
  if (!pyodideInitPromise) {
    pyodideInitPromise = (async () => {
      let lastErr = null;
      for (const indexURL of PYODIDE_INDEX_URLS) {
        const base = normalizePyodideIndexURL(indexURL);
        const scriptUrl = `${base}pyodide.js`;
        try {
          return await initPyodideOnce(indexURL);
        } catch (e) {
          lastErr = e;
          removePyodideScript(scriptUrl);
        }
      }
      const hint =
        "若你处于受限网络，可尝试：更换网络/VPN、使用可访问 unpkg 或 jsDelivr 的环境，或将 Pyodide 静态文件放到本地后自行改 PYODIDE_INDEX_URL。";
      throw new Error(
        `Pyodide 无法加载（已尝试 ${PYODIDE_INDEX_URLS.length} 个 CDN）。${lastErr ? `最后错误：${lastErr.message || lastErr}。` : ""} ${hint}`
      );
    })();
  }
  try {
    pyodideReady = await pyodideInitPromise;
    return pyodideReady;
  } catch (e) {
    pyodideInitPromise = null;
    throw e;
  }
}

async function getProtocolModels() {
  if (protocolModelsCache) return protocolModelsCache;
  const files = [
    "protocol_models_lr_subject.json",
    "protocol_models_mlp_subject.json",
    "protocol_models_lr_fold.json",
    "protocol_models_mlp_fold.json",
  ];
  const loaded = await Promise.all(
    files.map(async (f) => {
      const resp = await fetch(`./assets/${f}`);
      if (!resp.ok) {
        throw new Error(`无法加载 ${f}，请先运行协议模型导出脚本。`);
      }
      return [f, await resp.json()];
    })
  );
  protocolModelsCache = Object.fromEntries(loaded);
  return protocolModelsCache;
}

async function getGlobalWebModel() {
  if (globalWebModelCache) {
    return { ok: true, model: globalWebModelCache };
  }
  const resp = await fetch("./assets/web_model.json");
  if (!resp.ok) {
    return {
      ok: false,
      error: `无法加载 web_model.json（HTTP ${resp.status}），请先运行 python src/ml/export_web_model.py。`,
    };
  }
  const model = await resp.json();
  globalWebModelCache = model;
  return { ok: true, model };
}

function shannonEntropy(probs) {
  const eps = 1e-12;
  let h = 0;
  for (const p of probs) {
    if (p > eps) h -= p * Math.log(p);
  }
  return h;
}

function labelDistributionDiff(rows, labels) {
  const n = rows.length;
  const k = labels.length;
  const zeros = () => new Array(k).fill(0);
  if (!n) {
    return {
      p_true: Object.fromEntries(labels.map((l) => [l, 0])),
      p_pred: Object.fromEntries(labels.map((l) => [l, 0])),
      counts_true: Object.fromEntries(labels.map((l) => [l, 0])),
      counts_pred: Object.fromEntries(labels.map((l) => [l, 0])),
      tvd: 0,
      entropy_true: 0,
      entropy_pred: 0,
      entropy_delta: 0,
    };
  }
  const ct = zeros();
  const cp = zeros();
  for (const r of rows) {
    ct[r.y_true] += 1;
    cp[r.y_pred] += 1;
  }
  const pt = ct.map((c) => c / n);
  const pp = cp.map((c) => c / n);
  let tvd = 0;
  for (let i = 0; i < k; i++) tvd += Math.abs(pt[i] - pp[i]);
  tvd *= 0.5;
  return {
    p_true: Object.fromEntries(labels.map((l, i) => [l, pt[i]])),
    p_pred: Object.fromEntries(labels.map((l, i) => [l, pp[i]])),
    counts_true: Object.fromEntries(labels.map((l, i) => [l, ct[i]])),
    counts_pred: Object.fromEntries(labels.map((l, i) => [l, cp[i]])),
    tvd,
    entropy_true: shannonEntropy(pt),
    entropy_pred: shannonEntropy(pp),
    entropy_delta: shannonEntropy(pp) - shannonEntropy(pt),
  };
}

function evaluateGlobalLR(subjectData, model, labels) {
  const allRows = [];
  let expected = 0;
  for (const data of Object.values(subjectData)) {
    for (const tStr of Object.keys(data.featureByTrial)) {
      const t = Number(tStr);
      if (t in data.truthByTrial) expected += 1;
    }
  }
  for (const [subject, data] of Object.entries(subjectData)) {
    for (const tStr of Object.keys(data.featureByTrial)) {
      const t = Number(tStr);
      if (!(t in data.truthByTrial)) continue;
      const pred = predictLR(data.featureByTrial[t], model);
      allRows.push({
        subject: Number(subject),
        trial: t,
        y_true: data.truthByTrial[t],
        y_pred: pred.labelIndex,
        pred_label: labels[pred.labelIndex],
        confidence: pred.confidence,
      });
    }
  }
  const valid = allRows.length;
  const perSubject = [];
  for (const subject of Object.keys(subjectData)) {
    const sid = Number(subject);
    const rows = allRows.filter((r) => r.subject === sid);
    if (rows.length) {
      const m = computeMetrics(rows, labels.length);
      perSubject.push({
        subject: sid,
        accuracy: m.accuracy,
        macro_f1: m.macro_f1,
        valid_trials: m.valid_trials,
      });
    }
  }
  perSubject.sort((a, b) => a.subject - b.subject);
  const overall = computeMetrics(allRows, labels.length);
  return {
    model: "lr",
    protocol: "full_manifest",
    coverage: { expected_trials: expected, valid_trials: valid, ratio: expected ? valid / expected : 0 },
    overall,
    per_subject: perSubject,
    rows: allRows,
    distribution_diff: labelDistributionDiff(allRows, labels),
  };
}

function softmax(logits) {
  const maxVal = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxVal));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / s);
}

function standardize(vec, scaler) {
  return vec.map((v, i) => (v - scaler.mean[i]) / (scaler.scale[i] || 1.0));
}

function predictLR(vec, modelEntry) {
  const x = standardize(vec, modelEntry.scaler);
  const coef = modelEntry.classifier.coef;
  const intercept = modelEntry.classifier.intercept;
  const classes = modelEntry.classifier.classes;
  const logits = coef.map((row, cidx) => {
    let z = intercept[cidx];
    for (let i = 0; i < row.length; i++) z += row[i] * x[i];
    return z;
  });
  const probs = softmax(logits);
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestIdx]) bestIdx = i;
  return { labelIndex: classes[bestIdx], confidence: probs[bestIdx], probabilities: probs };
}

function linearForward(x, layer) {
  const w = layer.weight;
  const b = layer.bias;
  const out = new Array(w.length).fill(0);
  for (let o = 0; o < w.length; o++) {
    let v = b[o];
    const row = w[o];
    for (let i = 0; i < row.length; i++) v += row[i] * x[i];
    out[o] = v;
  }
  return out;
}

function relu(arr) {
  return arr.map((v) => (v > 0 ? v : 0));
}

function predictMLP(vec, modelEntry) {
  let x = standardize(vec, modelEntry.scaler);
  const layers = modelEntry.network.layers;
  x = relu(linearForward(x, layers[0]));
  x = relu(linearForward(x, layers[1]));
  const logits = linearForward(x, layers[2]);
  const probs = softmax(logits);
  const classes = modelEntry.classes;
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestIdx]) bestIdx = i;
  return { labelIndex: classes[bestIdx], confidence: probs[bestIdx], probabilities: probs };
}

function parseEmotionFromMoviePath(moviePath) {
  const normalized = moviePath.replaceAll("\\", "/");
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length < 2) return null;
  return parts[parts.length - 2].toLowerCase();
}

function parseSaveInfoCsv(csvText) {
  const lines = csvText.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  const emotions = [];
  for (const line of lines) {
    const cols = line.split(",");
    if (cols.length < 2) continue;
    const e = parseEmotionFromMoviePath(cols[1]);
    if (e) emotions.push(e);
  }
  return emotions;
}

async function extractTrialFeaturesFromMatBytes(bytes, featureType) {
  const prefixMap = { de: "de_", de_lds: "de_LDS_", psd: "psd_" };
  const prefix = prefixMap[featureType];
  if (!prefix) throw new Error(`不支持的 featureType: ${featureType}`);
  const py = await getPyodide();
  await ensurePyodideNumericLibs(py);
  py.globals.set("file_bytes", bytes);
  py.globals.set("feature_prefix", prefix);
  const pyCode = `
import io, json, numpy as np
import scipy.io as sio
mat = sio.loadmat(io.BytesIO(bytes(file_bytes.to_py())))
trials = []
for k, v in mat.items():
    if not k.startswith(feature_prefix):
        continue
    try:
        trial_idx = int(k.split("_")[-1])
    except Exception:
        continue
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim != 3:
        continue
    vec = arr.mean(axis=0).reshape(-1)
    trials.append((trial_idx, vec.tolist()))
trials = sorted(trials, key=lambda x: x[0])
json.dumps({"trial_indices": [t for t, _ in trials], "features": [f for _, f in trials]})
  `;
  const out = await py.runPythonAsync(pyCode);
  return JSON.parse(out);
}

function initConfusion(numClasses) {
  return new Array(numClasses).fill(0).map(() => new Array(numClasses).fill(0));
}

function computeMetrics(rows, numClasses) {
  if (!rows.length) {
    return {
      accuracy: 0,
      macro_f1: 0,
      confusion: initConfusion(numClasses),
      valid_trials: 0,
    };
  }
  const conf = initConfusion(numClasses);
  let correct = 0;
  for (const r of rows) {
    conf[r.y_true][r.y_pred] += 1;
    if (r.y_true === r.y_pred) correct += 1;
  }
  const accuracy = correct / rows.length;
  const eps = 1e-12;
  const f1s = [];
  for (let c = 0; c < numClasses; c++) {
    const tp = conf[c][c];
    let fp = 0;
    let fn = 0;
    for (let i = 0; i < numClasses; i++) {
      if (i !== c) {
        fp += conf[i][c];
        fn += conf[c][i];
      }
    }
    const p = tp / (tp + fp + eps);
    const r = tp / (tp + fn + eps);
    f1s.push((2 * p * r) / (p + r + eps));
  }
  return {
    accuracy,
    macro_f1: f1s.reduce((a, b) => a + b, 0) / numClasses,
    confusion: conf,
    valid_trials: rows.length,
  };
}

function summarizePredictions(rows, labels) {
  const counts = Object.fromEntries(labels.map((l) => [l, 0]));
  const confidences = [];
  const confByLabel = Object.fromEntries(labels.map((l) => [l, []]));
  for (const r of rows) {
    counts[r.pred_label] += 1;
    confidences.push(r.confidence);
    confByLabel[r.pred_label].push(r.confidence);
  }
  confidences.sort((a, b) => a - b);
  const mean = confidences.length ? confidences.reduce((a, b) => a + b, 0) / confidences.length : 0;
  const median =
    confidences.length === 0
      ? 0
      : confidences.length % 2
        ? confidences[(confidences.length - 1) / 2]
        : (confidences[confidences.length / 2 - 1] + confidences[confidences.length / 2]) / 2;
  const perLabelMean = {};
  for (const l of labels) {
    const arr = confByLabel[l];
    perLabelMean[l] = arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  }
  const topTrials = rows
    .slice()
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 15)
    .map((r) => ({ trial: r.trial, subject: r.subject, pred_label: r.pred_label, confidence: r.confidence }));
  const dominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  return {
    counts,
    dominant_label: dominant ? dominant[0] : "unknown",
    confidence_stats: {
      mean,
      median,
      min: confidences[0] ?? 0,
      max: confidences[confidences.length - 1] ?? 0,
    },
    per_label_confidence_mean: perLabelMean,
    top_trials_by_confidence: topTrials,
  };
}

function buildSubjectData(featurePayloadBySubject, emotionsBySubjectSession, labelToIdx) {
  const out = {};
  for (const [subject, featPayload] of Object.entries(featurePayloadBySubject)) {
    const featureByTrial = {};
    featPayload.trial_indices.forEach((t, i) => {
      featureByTrial[Number(t)] = featPayload.features[i];
    });

    const truthByTrial = {};
    const sessMap = emotionsBySubjectSession[subject] || {};
    for (let session = 1; session <= 4; session++) {
      const emotions = sessMap[String(session)] || [];
      for (let i = 0; i < emotions.length; i++) {
        const trial = (session - 1) * 20 + (i + 1);
        const label = emotions[i];
        if (label in labelToIdx) {
          truthByTrial[trial] = labelToIdx[label];
        }
      }
    }
    out[subject] = { featureByTrial, truthByTrial };
  }
  return out;
}

function evaluateSubjectDependent(subjectData, modelPack, labels, predictorName) {
  const allRows = [];
  const perSubject = [];
  let expected = 0;
  let valid = 0;
  for (const [subject, data] of Object.entries(subjectData)) {
    const model = modelPack.subjects[subject];
    if (!model) continue;
    const testTrials = new Set((model.test_trials || []).map((v) => Number(v)));
    const rows = [];
    for (const t of testTrials) {
      if (!(t in data.featureByTrial)) continue;
      expected += 1;
      if (!(t in data.truthByTrial)) continue;
      const pred =
        predictorName === "lr" ? predictLR(data.featureByTrial[t], model) : predictMLP(data.featureByTrial[t], model);
      rows.push({
        subject: Number(subject),
        trial: t,
        y_true: data.truthByTrial[t],
        y_pred: pred.labelIndex,
        pred_label: labels[pred.labelIndex],
        confidence: pred.confidence,
      });
    }
    valid += rows.length;
    if (rows.length) {
      const m = computeMetrics(rows, labels.length);
      perSubject.push({
        subject: Number(subject),
        accuracy: m.accuracy,
        macro_f1: m.macro_f1,
        valid_trials: m.valid_trials,
      });
    }
    allRows.push(...rows);
  }
  const overall = computeMetrics(allRows, labels.length);
  return {
    model: predictorName,
    protocol: "subject_dependent",
    coverage: { expected_trials: expected, valid_trials: valid, ratio: expected ? valid / expected : 0 },
    overall,
    per_subject: perSubject.sort((a, b) => a.subject - b.subject),
    rows: allRows,
  };
}

function evaluateSubjectIndependent(subjectData, modelPack, labels, predictorName) {
  const allRows = [];
  const perFold = [];
  let expected = 0;
  let valid = 0;
  for (const fold of modelPack.folds || []) {
    const testSubs = new Set((fold.test_subjects || []).map((v) => Number(v)));
    const foldRows = [];
    for (const [subject, data] of Object.entries(subjectData)) {
      if (!testSubs.has(Number(subject))) continue;
      for (const [trialStr, feat] of Object.entries(data.featureByTrial)) {
        const t = Number(trialStr);
        expected += 1;
        if (!(t in data.truthByTrial)) continue;
        const pred = predictorName === "lr" ? predictLR(feat, fold) : predictMLP(feat, fold);
        foldRows.push({
          subject: Number(subject),
          trial: t,
          y_true: data.truthByTrial[t],
          y_pred: pred.labelIndex,
          pred_label: labels[pred.labelIndex],
          confidence: pred.confidence,
        });
      }
    }
    valid += foldRows.length;
    const fm = computeMetrics(foldRows, labels.length);
    perFold.push({
      fold: fold.fold,
      test_subjects: fold.test_subjects,
      accuracy: fm.accuracy,
      macro_f1: fm.macro_f1,
      valid_trials: fm.valid_trials,
    });
    allRows.push(...foldRows);
  }
  const overall = computeMetrics(allRows, labels.length);
  return {
    model: predictorName,
    protocol: "subject_independent",
    coverage: { expected_trials: expected, valid_trials: valid, ratio: expected ? valid / expected : 0 },
    overall,
    per_fold: perFold,
    rows: allRows,
  };
}

function parseSubjectFromMatPath(path) {
  const m = path.match(/^EEF_features\/(\d+)\.mat$/);
  return m ? m[1] : null;
}

function parseSaveInfoPath(path) {
  const m = path.match(/^save_info\/(\d+)_\d{8}_(\d+)_save_info\.csv$/);
  if (!m) return null;
  return { subject: m[1], session: m[2] };
}

async function readZipUpload(zipFile, featureType) {
  if (!window.JSZip) {
    throw new Error("JSZip 未加载，请检查网络。");
  }
  const zip = await window.JSZip.loadAsync(await zipFile.arrayBuffer());
  const matBytesBySubject = {};
  const emotionsBySubjectSession = {};
  const entries = Object.values(zip.files);
  for (const entry of entries) {
    if (entry.dir) continue;
    const path = entry.name.replaceAll("\\", "/");
    const matSubject = parseSubjectFromMatPath(path);
    if (matSubject) {
      matBytesBySubject[matSubject] = await entry.async("uint8array");
      continue;
    }
    const si = parseSaveInfoPath(path);
    if (si) {
      const csvText = await entry.async("string");
      const emotions = parseSaveInfoCsv(csvText);
      emotionsBySubjectSession[si.subject] = emotionsBySubjectSession[si.subject] || {};
      emotionsBySubjectSession[si.subject][si.session] = emotions;
    }
  }

  const featurePayloadBySubject = {};
  for (const [subject, bytes] of Object.entries(matBytesBySubject)) {
    featurePayloadBySubject[subject] = await extractTrialFeaturesFromMatBytes(bytes, featureType);
  }
  return { featurePayloadBySubject, emotionsBySubjectSession };
}

async function runUploadedZipAnalysis(zipFile, featureType) {
  const models = await getProtocolModels();
  const lrSubject = models["protocol_models_lr_subject.json"];
  const mlpSubject = models["protocol_models_mlp_subject.json"];
  const lrFold = models["protocol_models_lr_fold.json"];
  const mlpFold = models["protocol_models_mlp_fold.json"];
  const labels = lrSubject.labels;
  const labelToIdx = Object.fromEntries(labels.map((l, idx) => [l, idx]));

  const { featurePayloadBySubject, emotionsBySubjectSession } = await readZipUpload(zipFile, featureType);
  const subjectData = buildSubjectData(featurePayloadBySubject, emotionsBySubjectSession, labelToIdx);
  const uploadedSubjects = Object.keys(subjectData).map((s) => Number(s)).sort((a, b) => a - b);

  const sdLR = evaluateSubjectDependent(subjectData, lrSubject, labels, "lr");
  const sdMLP = evaluateSubjectDependent(subjectData, mlpSubject, labels, "mlp");
  const siLR = evaluateSubjectIndependent(subjectData, lrFold, labels, "lr");
  const siMLP = evaluateSubjectIndependent(subjectData, mlpFold, labels, "mlp");

  const refRows = siLR.rows.length ? siLR.rows : sdLR.rows;
  const predSummary = summarizePredictions(refRows, labels);

  let global_full_data_lr = null;
  let global_model_skipped_reason = null;
  const globalRes = await getGlobalWebModel();
  if (!globalRes.ok) {
    global_model_skipped_reason = globalRes.error;
  } else {
    const wm = globalRes.model;
    if (wm.feature_type !== featureType) {
      global_model_skipped_reason = `全量网页模型特征为「${wm.feature_type}」，与当前选择「${featureType}」不一致；请切换特征类型或重新导出匹配的 web_model.json。`;
    } else if (JSON.stringify(wm.labels) !== JSON.stringify(labels)) {
      global_model_skipped_reason = "web_model.json 与协议模型的标签列表不一致，已跳过全量模型评估。";
    } else {
      global_full_data_lr = evaluateGlobalLR(subjectData, wm, labels);
    }
  }

  return {
    labels,
    feature_type_used: featureType,
    uploaded_subjects: uploadedSubjects,
    uploaded_subjects_count: uploadedSubjects.length,
    protocols: {
      subject_dependent: { lr: sdLR, mlp: sdMLP },
      subject_independent: { lr: siLR, mlp: siMLP },
    },
    global_full_data_lr,
    global_model_skipped_reason,
    ...predSummary,
    total_trials: refRows.length,
  };
}

window.UploadInference = {
  runUploadedZipAnalysis,
};

