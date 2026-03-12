/**
 * Qwen-Omni 情绪理解仪表盘 — WebSocket 客户端 + Chart.js 趋势图
 */

const EMOTION_LABELS = {
  happy: "开心", sad: "悲伤", angry: "愤怒", fearful: "恐惧",
  surprised: "惊讶", disgusted: "厌恶", neutral: "平静", contemptuous: "轻蔑"
};

const EMOTION_COLORS = {
  happy: "#fbbf24", sad: "#4a9eff", angry: "#f87171", fearful: "#a78bfa",
  surprised: "#f472b6", disgusted: "#a3e635", neutral: "#8b8fa3", contemptuous: "#fb923c"
};

// ── WebSocket 连接 ─────────────────────────────────────────

let ws = null;
let reconnectTimer = null;
const WS_URL = `ws://${location.host}/ws`;
let historyFrozen = false;

function connectWebSocket() {
  if (ws && ws.readyState <= WebSocket.OPEN) return;

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    document.getElementById("ws-status").textContent = "已连接";
    document.getElementById("ws-status").className = "status-value status-ok";
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateStatusBar(data.metrics);
      if (!historyFrozen) {
        updateEmotionCards(data.history);
      }
      updateTrendChart(data.trends);
    } catch (e) {
      console.error("WebSocket 消息解析失败:", e);
    }
  };

  ws.onclose = () => {
    document.getElementById("ws-status").textContent = "断开";
    document.getElementById("ws-status").className = "status-value status-warn";
    scheduleReconnect();
  };

  ws.onerror = () => {
    ws.close();
  };
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectWebSocket();
  }, 2000);
}

// ── 状态栏更新 ─────────────────────────────────────────────

function updateStatusBar(metrics) {
  if (!metrics) return;

  const latencyEl = document.getElementById("sys-latency");
  latencyEl.textContent = `${metrics.last_latency_ms} ms`;
  latencyEl.className = "status-value " + (metrics.within_budget ? "status-ok" : "status-error");

  document.getElementById("sys-avg-latency").textContent = `${metrics.avg_latency_ms} ms`;
  document.getElementById("sys-person-count").textContent = metrics.person_count;
  document.getElementById("sys-inference-count").textContent = metrics.inference_count;

  const statusEl = document.getElementById("sys-status");
  if (metrics.running === false) {
    statusEl.textContent = "已暂停";
    statusEl.className = "status-value status-warn";
  } else {
    statusEl.textContent = "运行中";
    statusEl.className = "status-value status-ok";
  }
}

// ── 情绪卡片渲染（推理历史） ─────────────────────────────────

function updateEmotionCards(history) {
  const container = document.getElementById("emotion-cards");
  if (!history || history.length === 0) {
    container.innerHTML = '<div class="empty-hint">等待推理结果...</div>';
    return;
  }

  const sortedHistory = history
    .slice()
    .sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
  const fragment = document.createDocumentFragment();

  for (const item of sortedHistory.reverse()) {
    const e = item;
    const card = document.createElement("div");
    card.className = "emotion-card";
    const hasIntensity = typeof e.emotion_intensity === "number";
    const hasConfidence = typeof e.confidence === "number";
    const intensityPct = hasIntensity ? Math.round(e.emotion_intensity * 100) : null;
    const confidencePct = hasConfidence ? Math.round(e.confidence * 100) : null;
    const primaryLabel = EMOTION_LABELS[e.primary_emotion] || e.primary_emotion;
    const secondaryLabel = e.secondary_emotion
      ? (EMOTION_LABELS[e.secondary_emotion] || e.secondary_emotion)
      : "无";

    const confidenceHtml = confidencePct != null
      ? `<span class="confidence-badge">置信度 ${confidencePct}%</span>`
      : "";
    const intensityHtml = intensityPct != null
      ? `<div class="intensity-bar-container">
          <span class="intensity-label">${intensityPct}%</span>
          <div class="intensity-bar-bg">
            <div class="intensity-bar-fill bar-${e.primary_emotion}"
                 style="width: ${intensityPct}%"></div>
          </div>
        </div>`
      : "";
    const descriptionHtml = e.description != null && e.description !== ""
      ? `<div class="description">${escapeHtml(e.description)}</div>`
      : "";
    const ts =
      typeof e.timestamp === "number"
        ? new Date(e.timestamp * 1000).toLocaleTimeString("zh-CN", {
            hour12: false,
          })
        : "";
    const frameCount = typeof e.frame_count === "number" ? e.frame_count : 0;
    const frameThumbs = [];
    for (let i = 0; i < frameCount; i++) {
      const frameUrl = `/api/history/frame/${encodeURIComponent(
        e.id
      )}/${i}`;
      frameThumbs.push(
        `<img class="history-frame" src="${frameUrl}" alt="帧 ${i + 1}" />`
      );
    }
    const framesHtml = frameThumbs.length
      ? `<div class="frame-strip">${frameThumbs.join("")}</div>`
      : "";
    const audioUrl = `/api/history/audio/${encodeURIComponent(e.id)}`;

    card.innerHTML = `
      <div class="card-header">
        <span class="person-id">${escapeHtml(
          e.person_id != null ? e.person_id : ""
        )}</span>
        <span class="timestamp">${escapeHtml(ts)}</span>
        ${confidenceHtml}
      </div>
      <div class="media-row">
        ${framesHtml}
        <audio class="history-audio" controls src="${audioUrl}"></audio>
      </div>
      <div class="primary-emotion emotion-${e.primary_emotion}">${primaryLabel}</div>
      <div class="secondary-emotion">次要情绪: ${secondaryLabel}</div>
      ${intensityHtml}
      ${descriptionHtml}
    `;
    fragment.appendChild(card);
  }

  container.innerHTML = "";
  container.appendChild(fragment);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ── Chart.js 趋势图 ───────────────────────────────────────

let trendChart = null;
const MAX_TREND_POINTS = 20;
const trendHistory = {};

function initChart() {
  const ctx = document.getElementById("trend-chart").getContext("2d");
  trendChart = new Chart(ctx, {
    type: "line",
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      scales: {
        x: {
          display: true,
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#8b8fa3", maxTicksLimit: 10 }
        },
        y: {
          min: 0, max: 1,
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#8b8fa3", stepSize: 0.2 },
          title: { display: true, text: "情绪强度", color: "#8b8fa3" }
        }
      },
      plugins: {
        legend: {
          labels: { color: "#e4e6f0", boxWidth: 12, padding: 16 }
        }
      },
      interaction: { mode: "index", intersect: false },
      elements: {
        point: { radius: 2, hoverRadius: 4 },
        line: { tension: 0.3 }
      }
    }
  });
}

function updateTrendChart(trends) {
  if (!trendChart || !trends) return;

  const now = new Date().toLocaleTimeString("zh-CN", { hour12: false });

  for (const [personId, dataPoints] of Object.entries(trends)) {
    if (!trendHistory[personId]) {
      trendHistory[personId] = [];
    }
    if (dataPoints.length > 0) {
      const latest = dataPoints[dataPoints.length - 1];
      if (typeof latest.emotion_intensity === "number") {
        trendHistory[personId].push({
          time: now,
          intensity: latest.emotion_intensity,
          emotion: latest.primary_emotion
        });
        if (trendHistory[personId].length > MAX_TREND_POINTS) {
          trendHistory[personId].shift();
        }
      }
    }
  }

  const allPersonIds = Object.keys(trendHistory).sort();
  if (allPersonIds.length === 0) return;

  const referenceId = allPersonIds[0];
  const labels = trendHistory[referenceId].map(p => p.time);

  const datasets = allPersonIds.map((pid, idx) => {
    const history = trendHistory[pid];
    const latestEmotion = history.length > 0 ? history[history.length - 1].emotion : "neutral";
    const color = EMOTION_COLORS[latestEmotion] || "#8b8fa3";
    return {
      label: pid,
      data: history.map(p => p.intensity),
      borderColor: color,
      backgroundColor: color + "33",
      fill: false,
      borderWidth: 2
    };
  });

  trendChart.data.labels = labels;
  trendChart.data.datasets = datasets;
  trendChart.update("none");
}

// ── 初始化 ─────────────────────────────────────────────────
async function sendPipelineCommand(action) {
  const endpoint =
    action === "start" ? "/api/pipeline/start" : "/api/pipeline/pause";
  try {
    const resp = await fetch(endpoint, { method: "POST" });
    if (!resp.ok) {
      console.error("控制流水线失败:", action, resp.status);
      return;
    }
    if (action === "pause") {
      historyFrozen = true;
    } else if (action === "start") {
      historyFrozen = false;
    }
  } catch (e) {
    console.error("控制流水线异常:", e);
  }
}

function bindControls() {
  const btnStart = document.getElementById("btn-start");
  const btnPause = document.getElementById("btn-pause");
  if (btnStart) {
    btnStart.addEventListener("click", () => sendPipelineCommand("start"));
  }
  if (btnPause) {
    btnPause.addEventListener("click", () => sendPipelineCommand("pause"));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initChart();
  connectWebSocket();
  bindControls();
});
