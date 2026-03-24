const state = {
  sessions: window.__INITIAL_STATE__.sessions || [],
  activeSession: window.__INITIAL_STATE__.activeSession || null,
  loading: false,
};

const chatListEl = document.getElementById("chat-list");
const chatCountBadgeEl = document.getElementById("chat-count-badge");
const chatTitleEl = document.getElementById("chat-title");
const messagesEl = document.getElementById("messages");
const chatFormEl = document.getElementById("chat-form");
const inputEl = document.getElementById("question-input");
const sendBtnEl = document.getElementById("send-btn");
const newChatBtnEl = document.getElementById("new-chat-btn");

function formatDate(value) {
  const date = new Date(value);
  return new Intl.DateTimeFormat("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "2-digit",
  }).format(date);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderChatList() {
  chatListEl.innerHTML = "";
  chatCountBadgeEl.textContent = String(state.sessions.length);

  for (const session of state.sessions) {
    const card = document.createElement("article");
    card.className = "history-card";
    if (state.activeSession && session.id === state.activeSession.id) {
      card.classList.add("active");
    }

    card.innerHTML = `
      <div class="history-card-top">
        <button class="history-open-btn" type="button" data-open-id="${session.id}">
          <p class="history-title">${escapeHtml(session.title)}</p>
          <p class="history-meta">Cập nhật ${formatDate(session.updated_at)}</p>
        </button>
        <button class="delete-btn" type="button" title="Xóa cuộc trò chuyện" data-delete-id="${session.id}">×</button>
      </div>
    `;
    chatListEl.appendChild(card);
  }
}

function renderEmptyState() {
  messagesEl.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">💬</div>
      <strong>Sẵn sàng tư vấn tuyển sinh</strong>
      <div>Hãy đặt câu hỏi về chỉ tiêu, phương thức xét tuyển, điểm chuẩn hoặc thông tin liên hệ của Đại học Nha Trang.</div>
    </div>
  `;
}

function renderMessages() {
  if (!state.activeSession) {
    chatTitleEl.textContent = "Cuộc trò chuyện mới";
    renderEmptyState();
    return;
  }

  chatTitleEl.textContent = state.activeSession.title;

  if (!state.activeSession.messages.length) {
    renderEmptyState();
    return;
  }

  messagesEl.innerHTML = state.activeSession.messages
    .map(
      (message) => `
        <div class="message-row ${message.role}">
          <div class="message-card">
            <p class="message-label">${message.role === "user" ? "Bạn" : "Chatbot"}</p>
            <p class="message-text">${escapeHtml(message.content)}</p>
          </div>
        </div>
      `,
    )
    .join("");

  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setLoading(isLoading) {
  state.loading = isLoading;
  chatFormEl.classList.toggle("loading", isLoading);
  inputEl.disabled = isLoading;
  sendBtnEl.disabled = isLoading;
  newChatBtnEl.disabled = isLoading;
}

function appendUserBubble(question) {
  const empty = messagesEl.querySelector(".empty-state");
  if (empty) {
    empty.remove();
  }

  const row = document.createElement("div");
  row.className = "message-row user";
  row.innerHTML = `
    <div class="message-card">
      <p class="message-label">Bạn</p>
      <p class="message-text">${escapeHtml(question)}</p>
    </div>
  `;
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendTypingBubble() {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.id = "typing-bubble";
  row.innerHTML = `
    <div class="message-card typing-card">
      <p class="message-label">Chatbot</p>
      <div class="typing-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeTypingBubble() {
  document.getElementById("typing-bubble")?.remove();
}

async function refreshSession(sessionId) {
  const response = await fetch(`/api/chats/${sessionId}`);
  if (!response.ok) {
    throw new Error("Không tải được cuộc trò chuyện.");
  }

  const data = await response.json();
  state.activeSession = data.session;
  state.sessions = state.sessions.map((session) =>
    session.id === data.session.id ? data.session : session,
  );
  renderChatList();
  renderMessages();
}

async function createChat() {
  setLoading(true);
  try {
    const response = await fetch("/api/chats", { method: "POST" });
    const data = await response.json();
    state.activeSession = data.session;
    state.sessions.unshift(data.session);
    renderChatList();
    renderMessages();
    inputEl.focus();
  } finally {
    setLoading(false);
  }
}

async function deleteChat(sessionId) {
  setLoading(true);
  try {
    const response = await fetch(`/api/chats/${sessionId}`, { method: "DELETE" });
    const data = await response.json();
    state.sessions = data.sessions;
    state.activeSession = state.sessions[0] || null;
    if (state.activeSession) {
      await refreshSession(state.activeSession.id);
    } else {
      renderChatList();
      renderMessages();
    }
  } finally {
    setLoading(false);
  }
}

async function sendMessage(question) {
  if (!state.activeSession) {
    await createChat();
  }

  inputEl.value = "";
  appendUserBubble(question);
  appendTypingBubble();
  setLoading(true);

  try {
    const response = await fetch(`/api/chats/${state.activeSession.id}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Không gửi được câu hỏi.");
    }

    state.activeSession = data.session;
    const existingIndex = state.sessions.findIndex((session) => session.id === data.session.id);
    if (existingIndex >= 0) {
      state.sessions.splice(existingIndex, 1);
    }
    state.sessions.unshift(data.session);

    removeTypingBubble();
    renderChatList();
    renderMessages();
    inputEl.focus();
  } catch (error) {
    removeTypingBubble();
    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.innerHTML = `
      <div class="message-card error-card">
        <p class="message-label">Chatbot</p>
        <p class="message-text">⚠ ${escapeHtml(error.message)}</p>
      </div>
    `;
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } finally {
    setLoading(false);
  }
}

chatListEl.addEventListener("click", async (event) => {
  const openId = event.target.closest("[data-open-id]")?.dataset.openId;
  const deleteId = event.target.closest("[data-delete-id]")?.dataset.deleteId;

  if (deleteId) {
    await deleteChat(deleteId);
    return;
  }

  if (openId) {
    setLoading(true);
    try {
      await refreshSession(openId);
    } finally {
      setLoading(false);
    }
  }
});

newChatBtnEl.addEventListener("click", createChat);

chatFormEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = inputEl.value.trim();
  if (!question || state.loading) {
    return;
  }
  await sendMessage(question);
});

renderChatList();
renderMessages();
