const API_URL = "http://127.0.0.1:8000";

const questionInput = document.getElementById("question");
const topKInput = document.getElementById("topK");
const askBtn = document.getElementById("askBtn");
const retrieveBtn = document.getElementById("retrieveBtn");
const answerDiv = document.getElementById("answer");
const confidenceDiv = document.getElementById("confidence");
const sourcesDiv = document.getElementById("sources");
const statusDiv = document.getElementById("status");

askBtn.addEventListener("click", () => askBackend("/ask"));
retrieveBtn.addEventListener("click", () => askBackend("/retrieve"));

checkHealth();

async function checkHealth() {
  try {
    const response = await fetch(`${API_URL}/health`);
    if (!response.ok) {
      throw new Error("Health check failed");
    }
    statusDiv.textContent = "Backend OK";
    statusDiv.className = "status ok";
  } catch (error) {
    statusDiv.textContent = "Backend offline";
    statusDiv.className = "status error";
  }
}

async function askBackend(endpoint) {
  const question = questionInput.value.trim();
  const topK = Number(topKInput.value || 5);

  if (!question) {
    alert("Vui lòng nhập câu hỏi.");
    return;
  }

  setLoading(true, endpoint);

  try {
    const response = await fetch(`${API_URL}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        top_k: topK,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();

    if (endpoint === "/ask") {
      answerDiv.textContent = data.answer || "Không tìm thấy câu trả lời.";
      confidenceDiv.textContent = `Confidence: ${Number(data.confidence || 0).toFixed(4)}`;
    } else {
      answerDiv.textContent = "Retrieve only.";
      confidenceDiv.textContent = "";
    }

    renderSources(data.sources || []);
  } catch (error) {
    answerDiv.textContent = "Có lỗi khi gọi backend.";
    confidenceDiv.textContent = "";
    sourcesDiv.innerHTML = "";
    console.error(error);
  } finally {
    setLoading(false, endpoint);
    checkHealth();
  }
}

function setLoading(isLoading, endpoint) {
  askBtn.disabled = isLoading;
  retrieveBtn.disabled = isLoading;
  askBtn.textContent = isLoading && endpoint === "/ask" ? "Đang hỏi..." : "Ask";
  retrieveBtn.textContent = isLoading && endpoint === "/retrieve" ? "Đang tìm..." : "Retrieve";
  if (isLoading) {
    answerDiv.textContent = "Đang xử lý...";
    confidenceDiv.textContent = "";
    sourcesDiv.innerHTML = "";
  }
}

function renderSources(sources) {
  if (!sources.length) {
    sourcesDiv.innerHTML = '<div class="sources-empty">Không có source.</div>';
    return;
  }

  sourcesDiv.innerHTML = sources.map(renderSource).join("");
}

function renderSource(source, index) {
  const title = source.title || "Untitled";
  const url = source.url || "";
  const text = source.text || "";
  const score = Number(source.score || 0).toFixed(4);
  const category = source.category || "";
  const chunkId = source.chunk_id || "";

  return `
    <div class="source-item">
      <div class="source-title">${index + 1}. ${escapeHtml(title)}</div>
      <div class="source-meta">
        Score: ${score} | Category: ${escapeHtml(category)} | Chunk: ${escapeHtml(chunkId)}
      </div>
      ${
        url
          ? `<div class="source-url">
               <a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(url)}</a>
             </div>`
          : ""
      }
      <div class="source-text">
        ${escapeHtml(text.slice(0, 700))}${text.length > 700 ? "..." : ""}
      </div>
    </div>
  `;
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
