let pyodideReady = null;
let webModelCache = null;

function ensurePyodideScript() {
  if (window.loadPyodide) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js";
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Pyodide 脚本加载失败，请检查网络。"));
    document.head.appendChild(s);
  });
}

async function getPyodide() {
  if (pyodideReady) {
    return pyodideReady;
  }
  pyodideReady = (async () => {
    await ensurePyodideScript();
    const py = await window.loadPyodide();
    await py.loadPackage("scipy");
    return py;
  })();
  return pyodideReady;
}

async function getWebModel() {
  if (webModelCache) {
    return webModelCache;
  }
  const resp = await fetch("./assets/web_model.json");
  if (!resp.ok) {
    throw new Error("无法加载 web_model.json，请先运行导出脚本。");
  }
  webModelCache = await resp.json();
  return webModelCache;
}

function softmax(logits) {
  const maxVal = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxVal));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / s);
}

function predictOne(featureVec, model) {
  const mean = model.scaler.mean;
  const scale = model.scaler.scale;
  const coef = model.classifier.coef;
  const intercept = model.classifier.intercept;
  const classes = model.classifier.classes;

  const x = featureVec.map((v, i) => (v - mean[i]) / (scale[i] || 1.0));
  const logits = coef.map((row, cidx) => {
    let z = intercept[cidx];
    for (let i = 0; i < row.length; i++) {
      z += row[i] * x[i];
    }
    return z;
  });
  const probs = softmax(logits);
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) {
      bestIdx = i;
    }
  }
  const labelIdx = classes[bestIdx];
  return {
    labelIndex: labelIdx,
    confidence: probs[bestIdx],
    probabilities: probs,
  };
}

async function extractTrialFeaturesFromMat(file, featureType) {
  const prefixMap = {
    de: "de_",
    de_lds: "de_LDS_",
    psd: "psd_",
  };
  const prefix = prefixMap[featureType];
  if (!prefix) {
    throw new Error(`不支持的 featureType: ${featureType}`);
  }

  const bytes = new Uint8Array(await file.arrayBuffer());
  const py = await getPyodide();
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
json.dumps({
    "trial_indices": [t for t, _ in trials],
    "features": [f for _, f in trials]
})
  `;
  const jsonStr = await py.runPythonAsync(pyCode);
  const payload = JSON.parse(jsonStr);
  return payload;
}

async function runUploadedMatInference(file, featureType) {
  const [model, trialPayload] = await Promise.all([getWebModel(), extractTrialFeaturesFromMat(file, featureType)]);
  if (trialPayload.features.length === 0) {
    throw new Error("该 .mat 中未找到可用特征键，请确认 featureType 是否匹配。");
  }
  if (trialPayload.features[0].length !== model.input_dim) {
    throw new Error(
      `特征维度不匹配：上传特征=${trialPayload.features[0].length}, 模型输入=${model.input_dim}。`
    );
  }

  const labels = model.labels;
  const rows = trialPayload.features.map((vec, i) => {
    const pred = predictOne(vec, model);
    return {
      trial: trialPayload.trial_indices[i],
      pred_label: labels[pred.labelIndex],
      confidence: pred.confidence,
      probabilities: pred.probabilities,
    };
  });

  const counts = {};
  for (const label of labels) {
    counts[label] = 0;
  }
  for (const row of rows) {
    counts[row.pred_label] += 1;
  }

  const dominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];

  const confidences = rows.map((r) => r.confidence).slice().sort((a, b) => a - b);
  const meanConf = confidences.reduce((a, b) => a + b, 0) / Math.max(confidences.length, 1);
  const medianConf =
    confidences.length % 2 === 1
      ? confidences[(confidences.length - 1) / 2]
      : (confidences[confidences.length / 2 - 1] + confidences[confidences.length / 2]) / 2;
  const minConf = confidences[0] ?? 0;
  const maxConf = confidences[confidences.length - 1] ?? 0;

  const confByLabel = {};
  for (const label of labels) {
    confByLabel[label] = [];
  }
  for (const row of rows) {
    confByLabel[row.pred_label].push(row.confidence);
  }
  const perLabelMeanConf = {};
  for (const label of labels) {
    const arr = confByLabel[label];
    perLabelMeanConf[label] = arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  }

  const topTrials = rows
    .slice()
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 15)
    .map((r) => ({
      trial: r.trial,
      pred_label: r.pred_label,
      confidence: r.confidence,
    }));

  return {
    labels,
    rows,
    counts,
    dominant_label: dominant ? dominant[0] : "unknown",
    total_trials: rows.length,
    model_type: model.model_type,
    feature_type_used: featureType,
    confidence_stats: {
      mean: meanConf,
      median: medianConf,
      min: minConf,
      max: maxConf,
    },
    per_label_confidence_mean: perLabelMeanConf,
    top_trials_by_confidence: topTrials,
  };
}

window.UploadInference = {
  runUploadedMatInference,
};

