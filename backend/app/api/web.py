"""Minimal chat UI served by FastAPI itself (no separate frontend process).

This is a single inline HTML page that talks to /v1/chat/completions over SSE.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


CHAT_HTML = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>오토에버 클라우드솔루션팀 챗봇</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  /* iPhone / iMessage-inspired theme */
  :root {
    --bg: #f2f2f7;                /* iOS systemGroupedBackground */
    --panel: rgba(255,255,255,0.78); /* translucent navbar */
    --panel-solid: #ffffff;
    --panel2: #ffffff;
    --field: #e9e9eb;             /* iOS field/secondary fill */
    --text: #1c1c1e;              /* iOS label */
    --text-soft: #3c3c43;
    --muted: #8e8e93;             /* iOS secondaryLabel / tertiaryLabel mix */
    --separator: rgba(60,60,67,0.18);
    --accent: #007aff;            /* iOS systemBlue */
    --accent-pressed: #0062cc;
    --user-bg-1: #2af;            /* iMessage blue gradient stops */
    --user-bg-2: #007aff;
    --user-fg: #ffffff;
    --assistant-bg: #e9e9eb;      /* iMessage gray bubble */
    --assistant-fg: #1c1c1e;
    --error: #ff3b30;             /* iOS systemRed */
    --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.06);
  }
  /* Forced light theme — do not follow system dark mode. */
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font: 15px/1.45 -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro",
          "Segoe UI", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
    -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
    display: flex; flex-direction: column;
    letter-spacing: -0.01em;
  }
  /* iOS-style translucent navigation bar */
  header {
    padding: 10px 16px;
    background: var(--panel);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    backdrop-filter: saturate(180%) blur(20px);
    border-bottom: 0.5px solid var(--separator);
    display: flex; align-items: center; gap: 10px;
    flex-wrap: wrap;
    position: sticky; top: 0; z-index: 10;
  }
  header h1 {
    font-size: 17px; font-weight: 600; margin: 0;
    color: var(--text); letter-spacing: -0.02em;
  }
  header .meta { color: var(--muted); font-size: 12px; }
  header .spacer { flex: 1; }
  header input {
    background: var(--field); border: none;
    color: var(--text); padding: 7px 12px; border-radius: 999px;
    font: inherit; font-size: 13px; width: 220px;
  }
  header input::placeholder { color: var(--muted); }
  header input:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
  header button {
    background: transparent; color: var(--accent); border: none;
    padding: 6px 10px; border-radius: 8px; cursor: pointer;
    font: inherit; font-size: 14px; font-weight: 400;
  }
  header button:hover { background: rgba(0,122,255,0.08); }
  header button:active { background: rgba(0,122,255,0.16); }
  main {
    flex: 1; overflow-y: auto; padding: 16px 14px 24px;
    display: flex; flex-direction: column; gap: 4px;
    max-width: 760px; margin: 0 auto; width: 100%;
  }
  /* iMessage-style bubbles */
  .msg { display: flex; flex-direction: column; max-width: 78%; margin-top: 6px; }
  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.assistant { align-self: flex-start; align-items: flex-start; }
  .msg .bubble {
    padding: 8px 14px; border-radius: 18px;
    white-space: pre-wrap; word-wrap: break-word;
    font-size: 15px; line-height: 1.35;
    box-shadow: 0 1px 0.5px rgba(0,0,0,0.06);
  }
  .msg.user .bubble {
    background: linear-gradient(180deg, var(--user-bg-1), var(--user-bg-2));
    color: var(--user-fg);
    border-bottom-right-radius: 4px;
  }
  .msg.assistant .bubble {
    background: var(--assistant-bg); color: var(--assistant-fg);
    border-bottom-left-radius: 4px;
  }
  .msg .progress {
    color: var(--muted); font-size: 12px; line-height: 1.55;
    background: var(--field);
    border-radius: 16px; padding: 8px 14px; margin-bottom: 4px;
    white-space: pre-wrap;
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    border-bottom-left-radius: 4px;
    max-width: 100%;
  }
  .msg .body { margin-top: 0; }
  .msg .body a { color: var(--accent); text-decoration: none; }
  .msg .body a:hover { text-decoration: underline; }
  .msg .body a.cite {
    font-size: 90%; font-weight: 600; padding: 0 2px; border-radius: 4px;
    background: rgba(120,120,128,0.14);
  }
  .msg .body a.cite:hover { background: rgba(120,120,128,0.28); text-decoration: none; }
  .msg .body strong { color: var(--text); font-weight: 600; }
  .msg.user .body strong { color: #ffffff; }
  .msg .body code {
    background: rgba(120,120,128,0.18); padding: 1px 5px;
    border-radius: 5px; font-size: 90%;
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }
  .msg.user .body code { background: rgba(255,255,255,0.22); }
  .msg .body pre {
    background: rgba(120,120,128,0.12); padding: 10px;
    border-radius: 12px; overflow-x: auto;
    margin: 8px 0;
  }
  .msg .body pre code { background: none; padding: 0; font-size: 13px; }
  /* iOS-style input footer */
  footer {
    padding: 8px 12px calc(12px + env(safe-area-inset-bottom));
    background: var(--panel);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    backdrop-filter: saturate(180%) blur(20px);
    border-top: 0.5px solid var(--separator);
  }
  .input-wrap {
    max-width: 760px; margin: 0 auto;
    display: flex; gap: 8px; align-items: flex-end;
    background: var(--panel2);
    border: 1px solid var(--separator);
    border-radius: 22px;
    padding: 4px 4px 4px 14px;
  }
  textarea {
    flex: 1; resize: none; min-height: 32px; max-height: 200px;
    background: transparent; color: var(--text);
    border: none; padding: 8px 4px;
    font: inherit; font-size: 15px; line-height: 1.35;
  }
  textarea:focus { outline: none; }
  textarea::placeholder { color: var(--muted); }
  button.send {
    background: var(--accent); color: white; border: none;
    width: 34px; height: 34px; border-radius: 50%;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; padding: 0;
    transition: transform 0.06s ease, background 0.12s ease;
  }
  button.send:hover { background: var(--accent-pressed); }
  button.send:active { transform: scale(0.94); }
  button.send:disabled { opacity: 0.4; cursor: not-allowed; }
  button.send::before { content: "↑"; font-weight: 700; line-height: 1; }
  /* Empty state */
  .empty {
    color: var(--muted); text-align: center; padding: 80px 20px;
  }
  .empty h2 { font-size: 22px; font-weight: 700; margin: 0 0 8px; color: var(--text); letter-spacing: -0.02em; }
  .empty p { margin: 4px 0; font-size: 14px; }
  .empty code {
    background: var(--field); padding: 1px 6px; border-radius: 5px;
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 90%;
  }
  .err { color: var(--error); }

  /* Login screen — iOS modal card */
  #login-screen {
    flex: 1; display: flex; align-items: center; justify-content: center;
    padding: 20px;
    background: var(--bg);
  }
  #login-card {
    background: var(--panel-solid);
    border-radius: 18px; padding: 28px 24px;
    width: 100%; max-width: 360px;
    box-shadow: var(--shadow);
    border: 0.5px solid var(--separator);
  }
  #login-card h2 {
    margin: 0 0 6px; font-size: 22px; font-weight: 700;
    color: var(--text); text-align: center; letter-spacing: -0.02em;
  }
  #login-card p {
    margin: 0 0 22px; color: var(--muted);
    font-size: 13px; text-align: center; line-height: 1.4;
  }
  #login-card label {
    display: block; margin-bottom: 6px;
    color: var(--muted); font-size: 12px;
    font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em;
  }
  #login-card input {
    width: 100%; background: var(--field); color: var(--text);
    border: none; border-radius: 12px;
    padding: 12px 14px; font: inherit; font-size: 16px;
  }
  #login-card input:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
  #login-card button {
    margin-top: 16px; width: 100%; background: var(--accent);
    color: white; border: none; padding: 13px;
    border-radius: 12px; font: inherit; font-size: 16px;
    font-weight: 600; cursor: pointer;
    transition: background 0.12s ease, transform 0.06s ease;
  }
  #login-card button:hover { background: var(--accent-pressed); }
  #login-card button:active { transform: scale(0.985); }
  #login-card button:disabled { opacity: 0.5; cursor: not-allowed; }
  #login-card .hint {
    color: var(--error); font-size: 12px;
    margin-top: 10px; min-height: 16px; text-align: center;
  }

  /* iOS user pill in nav bar */
  header .user-pill {
    background: var(--field); color: var(--text);
    border: none;
    padding: 5px 11px; border-radius: 999px;
    font-size: 12px; font-weight: 500;
  }

  /* Mobile tweaks */
  @media (max-width: 520px) {
    header { padding: 10px 12px; gap: 8px; }
    header h1 { font-size: 16px; }
    header input { width: 140px; font-size: 12px; }
    header .meta { display: none; }
    main { padding: 12px 10px 18px; }
    .msg { max-width: 86%; }
  }
</style>
</head>
<body>
  <!-- Login screen: shown when ?user_id=<id> is missing from the URL -->
  <div id="login-screen" style="display:none;">
    <div id="login-card">
      <h2>오토에버 클라우드솔루션팀 챗봇</h2>
      <p>사용자 아이디를 입력하면 챗봇으로 진입합니다. 입력한 아이디는 URL 파라미터로 부착됩니다.</p>
      <form id="login-form" autocomplete="off">
        <label for="login-id">사용자 아이디</label>
        <input id="login-id" type="text" placeholder="아이디를 입력하세요" autocomplete="off" autofocus>
        <button type="submit">로그인</button>
        <div class="hint" id="login-hint"></div>
      </form>
    </div>
  </div>

  <!-- Chat UI: shown when ?user_id=<id> is present -->
  <header id="chat-header" style="display:none;">
    <h1>오토에버 클라우드솔루션팀 챗봇</h1>
    <span class="spacer"></span>
    <span class="user-pill" id="user-pill"></span>
    <button id="clear">대화 초기화</button>
    <button id="logout">로그아웃</button>
  </header>
  <main id="chat" style="display:none;">
    <div class="empty" id="empty"></div>
  </main>
  <footer id="chat-footer" style="display:none;">
    <div class="input-wrap">
      <textarea id="input" rows="1" placeholder="질문을 입력하세요" aria-label="질문 입력 (Enter 전송, Shift+Enter 줄바꿈)"></textarea>
      <button class="send" id="send" aria-label="전송"></button>
    </div>
  </footer>

<script>
(function() {
  // ── Login gate: require ?user_id=<id> in the URL ──
  const params = new URLSearchParams(window.location.search);
  const userId = (params.get('user_id') || '').trim();

  const loginScreen = document.getElementById('login-screen');
  const chatHeader = document.getElementById('chat-header');
  const chatFooter = document.getElementById('chat-footer');
  const chatMain = document.getElementById('chat');

  if (!userId) {
    loginScreen.style.display = 'flex';
    const form = document.getElementById('login-form');
    const idInput = document.getElementById('login-id');
    const hint = document.getElementById('login-hint');
    // Pre-fill with the last logged-in id, if any.
    try {
      const last = localStorage.getItem('rag-chat:lastUserId');
      if (last) idInput.value = last;
    } catch (e) {}
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const v = idInput.value.trim();
      if (!v) { hint.textContent = '아이디를 입력하세요.'; return; }
      if (!/^[A-Za-z0-9._-]{1,64}$/.test(v)) {
        hint.textContent = '영문/숫자/._- 만 사용 (최대 64자).';
        return;
      }
      try { localStorage.setItem('rag-chat:lastUserId', v); } catch (e) {}
      const url = new URL(window.location.href);
      url.searchParams.set('user_id', v);
      window.location.href = url.toString();
    });
    return; // Stop here — chat UI stays hidden until logged in.
  }

  // Show chat UI now that we have a user_id.
  chatHeader.style.display = '';
  chatFooter.style.display = '';
  chatMain.style.display = '';

  const chat = chatMain;
  const empty = document.getElementById('empty');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('send');
  const clearBtn = document.getElementById('clear');
  const logoutBtn = document.getElementById('logout');
  const userPill = document.getElementById('user-pill');
  userPill.textContent = '👤 ' + userId;

  // Per-user message storage so different ids don't collide.
  const STORAGE_KEY = 'rag-chat:messages:v2:' + userId;
  // Per-conversation session id (UUID). Reset on "대화 초기화" so debug-mode
  // questions only see turns from the current thread. Persists across reload.
  const SESSION_KEY = 'rag-chat:session:v1:' + userId;

  let messages = [];  // [{role, content}]
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) messages = JSON.parse(saved);
  } catch (e) { messages = []; }

  function newSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
      return window.crypto.randomUUID();
    }
    return 'sess-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
  }
  let sessionId;
  try {
    sessionId = localStorage.getItem(SESSION_KEY) || '';
  } catch (e) { sessionId = ''; }
  if (!sessionId) {
    sessionId = newSessionId();
    try { localStorage.setItem(SESSION_KEY, sessionId); } catch (e) {}
  }

  function autosize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';
  }
  input.addEventListener('input', autosize);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  sendBtn.addEventListener('click', send);
  clearBtn.addEventListener('click', () => {
    if (messages.length && !confirm('대화를 모두 지울까요?')) return;
    messages = [];
    localStorage.removeItem(STORAGE_KEY);
    // Start a new session — debug-mode follow-ups won't pull from prior thread.
    sessionId = newSessionId();
    try { localStorage.setItem(SESSION_KEY, sessionId); } catch (e) {}
    rerenderHistory();
  });
  logoutBtn.addEventListener('click', () => {
    if (!confirm('로그아웃 할까요? 현재 대화 기록은 다음 로그인 시 그대로 보입니다.')) return;
    const url = new URL(window.location.href);
    url.searchParams.delete('user_id');
    window.location.href = url.pathname + (url.search ? url.search : '');
  });

  function persist() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(messages)); } catch (e) {}
  }
  // The intro message is rendered as a persistent assistant bubble at the top
  // of every chat. It is NOT added to `messages` (so it never gets sent to the
  // backend as history) but stays visible above the conversation forever —
  // streamed (typewriter) on first appearance, static on subsequent reloads.
  const INTRO_TEXT = '안녕하세요 저는 오토에버 클라우드솔루션팀 챗봇입니다. 무엇을 도와드릴까요?';
  // Typewriter timer for the intro stream — must be cancellable so that if
  // the user sends a message mid-stream we don't keep ticking against a
  // detached DOM node.
  let introTimer = null;
  let introWrap = null;

  function cancelIntroStream() {
    if (introTimer) { clearTimeout(introTimer); introTimer = null; }
    introWrap = null;
  }

  // Force-finish any in-flight typewriter so the bubble is fully populated
  // before the user message lands beneath it. Keeps the intro bubble in the
  // DOM (does NOT remove it).
  function freezeIntro() {
    if (introTimer) { clearTimeout(introTimer); introTimer = null; }
    if (introWrap) {
      const b = introWrap.querySelector('.bubble.body');
      if (b) b.textContent = INTRO_TEXT;
    }
    introWrap = null;
  }

  function renderIntroStreaming() {
    cancelIntroStream();
    const wrap = document.createElement('div');
    wrap.className = 'msg assistant intro';
    const body = document.createElement('div');
    body.className = 'bubble body';
    wrap.appendChild(body);
    chat.appendChild(wrap);
    introWrap = wrap;
    // Typewriter stream: append one char per tick.
    let i = 0;
    const STEP_MS = 35;       // per-char delay
    const START_MS = 250;     // small initial pause so the bubble appears first
    const tick = () => {
      if (!introWrap) return; // cancelled
      if (i >= INTRO_TEXT.length) { introTimer = null; return; }
      body.textContent += INTRO_TEXT[i++];
      chat.scrollTop = chat.scrollHeight;
      introTimer = setTimeout(tick, STEP_MS);
    };
    introTimer = setTimeout(tick, START_MS);
  }

  function renderIntroStatic() {
    const wrap = document.createElement('div');
    wrap.className = 'msg assistant intro';
    const body = document.createElement('div');
    body.className = 'bubble body';
    body.textContent = INTRO_TEXT;
    wrap.appendChild(body);
    chat.appendChild(wrap);
  }

  function rerenderHistory() {
    cancelIntroStream();
    chat.innerHTML = '';
    if (!messages.length) {
      // Fresh chat — animate the greeting.
      renderIntroStreaming();
    } else {
      // Existing conversation — show the greeting instantly at the top,
      // then render messages below it. Greeting persists across turns.
      renderIntroStatic();
      for (const m of messages) renderMessage(m.role, m.content);
    }
  }

  // Tiny renderer: escape HTML, autolink http(s), **bold**, `code`, triple-backtick code blocks,
  // and inline citation tokens [N] mapped to source URLs via the CITES marker.
  function escapeHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }
  // Parse a trailing CITES marker out of the answer body. Returns the text
  // without the marker plus a {n: url} map so [N] tokens can be linked.
  function parseCites(text) {
    const m = text.match(/\\n?<!--CITES:(\\[[\\s\\S]*?\\])-->/);
    if (!m) return { stripped: text, cites: {} };
    let cites = {};
    try {
      const arr = JSON.parse(m[1]);
      for (const c of arr) {
        if (c && typeof c.n === 'number' && c.url) cites[c.n] = c.url;
      }
    } catch (e) { /* malformed marker — render as plain text */ }
    const stripped = text.slice(0, m.index) + text.slice(m.index + m[0].length);
    return { stripped, cites };
  }
  function renderText(text, cites) {
    cites = cites || {};
    // Pull out triple-backtick code blocks first.
    const blocks = [];
    text = text.replace(/```([\\s\\S]*?)```/g, (_, code) => {
      const i = blocks.length;
      blocks.push(code);
      return `\\u0000CODEBLOCK${i}\\u0000`;
    });
    let html = escapeHtml(text);
    html = html.replace(/`([^`]+)`/g, (_, c) => `<code>${escapeHtml(c)}</code>`);
    html = html.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
    html = html.replace(/(https?:\\/\\/[^\\s)]+)/g,
      (m) => `<a href="${m}" target="_blank" rel="noopener">${m}</a>`);
    // Replace inline [N] citation tokens with anchors when a URL is mapped.
    html = html.replace(/\\[(\\d+)\\]/g, (m, n) => {
      const url = cites[+n];
      if (!url) return m;
      return `<a class="cite" href="${url}" target="_blank" rel="noopener">${m}</a>`;
    });
    html = html.replace(/\\u0000CODEBLOCK(\\d+)\\u0000/g, (_, i) => {
      return `<pre><code>${escapeHtml(blocks[+i])}</code></pre>`;
    });
    return html;
  }

  function renderMessage(role, content) {
    // (intro is cleared by send() before the first message lands here)
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + role;
    if (role === 'assistant') {
      const progress = document.createElement('div');
      progress.className = 'progress';
      progress.style.display = 'none';
      const body = document.createElement('div');
      body.className = 'bubble body';
      const SEP = '\\n─────────────────────────────────────\\n';
      const idx = content.indexOf(SEP);
      if (idx >= 0) {
        progress.textContent = content.slice(0, idx).trim();
        const parsed = parseCites(content.slice(idx + SEP.length));
        body.innerHTML = renderText(parsed.stripped, parsed.cites);
        if (progress.textContent) progress.style.display = '';
      } else {
        progress.textContent = content.trim();
        if (progress.textContent) progress.style.display = '';
      }
      wrap.appendChild(progress);
      wrap.appendChild(body);
    } else {
      const body = document.createElement('div');
      body.className = 'bubble';
      body.textContent = content;
      wrap.appendChild(body);
    }
    chat.appendChild(wrap);
    return wrap;
  }

  function liveAssistantMessage() {
    // (intro is cleared by send() before the first message lands here)
    const wrap = document.createElement('div');
    wrap.className = 'msg assistant';
    const progress = document.createElement('div');
    progress.className = 'progress';
    progress.style.display = 'none';
    const body = document.createElement('div');
    body.className = 'bubble body';
    body.textContent = '…';
    wrap.appendChild(progress);
    wrap.appendChild(body);
    chat.appendChild(wrap);
    let buf = '';
    const SEP = '\\n─────────────────────────────────────\\n';
    function update(text) {
      buf += text;
      const idx = buf.indexOf(SEP);
      if (idx >= 0) {
        const before = buf.slice(0, idx).trim();
        const after = buf.slice(idx + SEP.length);
        progress.textContent = before;
        if (before) progress.style.display = '';
        const parsed = parseCites(after);
        body.innerHTML = renderText(parsed.stripped, parsed.cites);
      } else {
        progress.textContent = buf.trim();
        if (progress.textContent) progress.style.display = '';
        body.textContent = '';
      }
      chat.scrollTop = chat.scrollHeight;
    }
    function finalText() { return buf; }
    return { update, finalText, body };
  }

  rerenderHistory();
  chat.scrollTop = chat.scrollHeight;

  async function send() {
    const text = input.value.trim();
    if (!text) return;
    sendBtn.disabled = true;
    input.disabled = true;

    // First turn: freeze the streaming intro (don't remove it). The greeting
    // bubble stays at the top; the user message and answer land beneath it.
    if (messages.length === 0) freezeIntro();

    messages.push({ role: 'user', content: text });
    persist();
    renderMessage('user', text);
    input.value = '';
    autosize();
    chat.scrollTop = chat.scrollHeight;

    const live = liveAssistantMessage();
    chat.scrollTop = chat.scrollHeight;

    try {
      const headers = { 'Content-Type': 'application/json' };
      const resp = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: 'rag-chatbot',
          messages: messages,
          stream: true,
          user_id: userId,
          session_id: sessionId,
        }),
      });
      if (!resp.ok || !resp.body) {
        const errText = await resp.text();
        live.body.innerHTML = '<span class="err">요청 실패: ' + escapeHtml(errText) + '</span>';
        return;
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let pending = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        pending += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = pending.indexOf('\\n\\n')) >= 0) {
          const raw = pending.slice(0, idx).trim();
          pending = pending.slice(idx + 2);
          if (!raw.startsWith('data:')) continue;
          const data = raw.slice(5).trim();
          if (data === '[DONE]') continue;
          try {
            const json = JSON.parse(data);
            const delta = (json.choices && json.choices[0] && json.choices[0].delta) || {};
            if (delta.content) live.update(delta.content);
          } catch (e) { /* ignore parse glitches */ }
        }
      }
      messages.push({ role: 'assistant', content: live.finalText() });
      persist();
    } catch (e) {
      live.body.innerHTML = '<span class="err">네트워크 오류: ' + escapeHtml(String(e)) + '</span>';
    } finally {
      sendBtn.disabled = false;
      input.disabled = false;
      input.focus();
    }
  }
})();
</script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def chat_ui() -> HTMLResponse:
    return HTMLResponse(content=CHAT_HTML)
