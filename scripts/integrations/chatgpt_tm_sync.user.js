// ==UserScript==
// @name         TeamMemory — ChatGPT Session Distiller
// @namespace    https://github.com/team-memory/chatgpt-sync
// @version      1.0.0
// @description  Push active ChatGPT conversations to local TeamMemory Daemon (localhost:3901/ingest/session) for automatic knowledge distillation.
// @match        https://chatgpt.com/*
// @match        https://chat.openai.com/*
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// @connect      localhost
// ==/UserScript==

(function () {
  "use strict";

  const DAEMON_INGEST_URL = "http://127.0.0.1:3901/ingest/session";

  function extractDomMessages() {
    const turns = document.querySelectorAll("[data-message-author-role]");
    const messages = [];
    turns.forEach((el) => {
      const role = el.getAttribute("data-message-author-role") || "user";
      const text = (el.innerText || "").trim();
      if (text) {
        messages.push({
          role: role === "assistant" ? "assistant" : "user",
          content: text,
        });
      }
    });
    return messages;
  }

  function getConversationId() {
    const match = window.location.pathname.match(/\/c\/([a-zA-Z0-9-]+)/);
    return match ? match[1] : "web-" + Date.now();
  }

  function syncToTeamMemory(btn) {
    const messages = extractDomMessages();
    if (messages.length < 2) {
      btn.textContent = "TM: Too short";
      setTimeout(() => (btn.textContent = "Distill to TM"), 2000);
      return;
    }
    const payload = {
      source_platform: "chatgpt",
      conversation_id: "chatgpt-" + getConversationId(),
      title: document.title.replace(" - ChatGPT", "").trim(),
      messages: messages,
    };

    btn.textContent = "TM: Distilling...";
    GM_xmlhttpRequest({
      method: "POST",
      url: DAEMON_INGEST_URL,
      headers: { "Content-Type": "application/json" },
      data: JSON.stringify(payload),
      onload: function (resp) {
        if (resp.status >= 200 && resp.status < 300) {
          btn.textContent = "TM: Queued ✓";
        } else {
          btn.textContent = "TM: Filtered/Skipped";
        }
        setTimeout(() => (btn.textContent = "Distill to TM"), 3000);
      },
      onerror: function () {
        btn.textContent = "TM: Daemon Offline";
        setTimeout(() => (btn.textContent = "Distill to TM"), 3000);
      },
    });
  }

  function mountButton() {
    if (document.getElementById("tm-distill-btn")) return;
    const btn = document.createElement("button");
    btn.id = "tm-distill-btn";
    btn.textContent = "Distill to TM";
    btn.style.cssText =
      "position:fixed;bottom:20px;right:20px;z-index:9999;padding:8px 14px;" +
      "border-radius:8px;background:#10a37f;color:#fff;font-size:13px;" +
      "font-weight:600;border:none;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,0.2);";
    btn.addEventListener("click", () => syncToTeamMemory(btn));
    document.body.appendChild(btn);
  }

  setInterval(mountButton, 3000);
})();
