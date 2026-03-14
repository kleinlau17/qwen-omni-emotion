/**
 * Qwen-Omni 仪表盘 — WebSocket 客户端
 */

// ── WebSocket 连接 ─────────────────────────────────────────

let ws = null;
let reconnectTimer = null;
const WS_URL = `ws://${location.host}/ws`;
let historyFrozen = false;
const ACTION_LABELS = {
  idle: "保持不动",
  "neutral.affirm.playful.low": "俏皮点头",
  "neutral.affirm.default.mid": "正常点头",
  "neutral.affirm.default.high": "用力点头",
  "neutral.affirm.expressive.high": "激动认同",
  "shy.affirm.inquiring.mid": "害羞点头",
  "angry.affirm.default.high": "怒气认同",
  "sad.affirm.default.low": "低落认同",
  "neutral.deny.quick.low": "快速摇头",
  "neutral.deny.default.mid": "正常摇头",
  "shy.deny.inquiring.mid": "害羞摇头",
  "angry.deny.default.high": "怒气摇头",
  "sad.deny.default.low": "伤心摇头",
  "neutral.attention.default.mid": "认真注视",
  "neutral.recover.default.mid": "鼓励",
  "neutral.dialogue.default.mid": "对话回应",
  "angry.greet.default.mid": "不耐烦",
  "neutral.think.murmur.low": "小声嘀咕",
  "neutral.think.muted.low": "安静沉思",
  "neutral.think.animated.mid": "活跃地思索",
  "neutral.question.default.low": "轻声提问",
  "neutral.question.default.mid": "正常提问",
  "neutral.question.default.high": "大声提问",
  "neutral.surprise.quick.low": "轻微惊讶",
  "neutral.alarm.expressive.high": "惊吓尖叫",
  "neutral.apology.default.low": "抱歉",
  "sad.sigh.default.low": "叹气",
};

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
    const actionText = ACTION_LABELS[e.action] || "未定义动作";
    const card = document.createElement("div");
    card.className = "emotion-card";
    const reasonTextRaw = e.reason ?? e.description ?? "";
    const reasonText = String(reasonTextRaw).trim();
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
      </div>
      <div class="media-row">
        ${framesHtml}
        <audio class="history-audio" controls src="${audioUrl}"></audio>
      </div>
      <div class="action-text">交互动作: ${escapeHtml(actionText)}</div>
      <div class="reason-text">动作原因: ${escapeHtml(reasonText || "-")}</div>
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
  connectWebSocket();
  bindControls();
});
