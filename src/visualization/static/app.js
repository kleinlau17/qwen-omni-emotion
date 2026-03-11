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
      updateEmotionCards(data.emotions);
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
}

// ── 情绪卡片渲染 ───────────────────────────────────────────

function updateEmotionCards(emotions) {
  const container = document.getElementById("emotion-cards");
  if (!emotions || Object.keys(emotions).length === 0) {
    container.innerHTML = '<div class="empty-hint">等待推理结果...</div>';
    return;
  }

  const sortedIds = Object.keys(emotions).sort();
  const fragment = document.createDocumentFragment();

  for (const personId of sortedIds) {
    const e = emotions[personId];
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

    card.innerHTML = `
      <div class="card-header">
        <span class="person-id">${escapeHtml(e.person_id != null ? e.person_id : personId)}</span>
        ${confidenceHtml}
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

document.addEventListener("DOMContentLoaded", () => {
  initChart();
  connectWebSocket();
});
