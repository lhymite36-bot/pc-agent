"""Localhost chat UI with Memory view, voice picker, and settings."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import json as _json
from pydantic import BaseModel

if TYPE_CHECKING:
    from pc_agent.core.memory import MemoryStore
    from pc_agent.core.voice import VoiceEngine

logger = logging.getLogger("pc_agent.web_ui")

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PC Agent</title>
<style>
  :root { --bg:#0f1115; --panel:#1a1f29; --ink:#e8ecf1; --muted:#8b95a8; --accent:#6aa7ff; --danger:#ff6b6b; --ok:#3dd68c; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.45 system-ui,Segoe UI,sans-serif; background:var(--bg); color:var(--ink); height:100vh; display:flex; }
  nav { width:200px; background:var(--panel); padding:16px; display:flex; flex-direction:column; gap:8px; }
  nav h1 { font-size:16px; margin:0 0 12px; }
  nav button { background:transparent; border:1px solid #2a3344; color:var(--ink); padding:10px 12px; border-radius:8px; text-align:left; cursor:pointer; }
  nav button.active, nav button:hover { border-color:var(--accent); color:var(--accent); }
  main { flex:1; display:flex; flex-direction:column; min-width:0; }
  .view { display:none; flex:1; flex-direction:column; min-height:0; }
  .view.active { display:flex; }
  #chat-log { flex:1; overflow:auto; padding:20px; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:720px; padding:12px 14px; border-radius:12px; white-space:pre-wrap; }
  .msg.user { background:#243044; align-self:flex-end; }
  .msg.assistant { background:var(--panel); align-self:flex-start; border:1px solid #2a3344; }
  .composer { display:flex; gap:8px; padding:16px; border-top:1px solid #2a3344; }
  .composer input, .composer textarea { flex:1; background:#12161e; border:1px solid #2a3344; color:var(--ink); border-radius:10px; padding:12px; }
  .composer button, .btn { background:var(--accent); color:#061018; border:0; border-radius:10px; padding:10px 16px; font-weight:600; cursor:pointer; }
  .btn.danger { background:var(--danger); color:#fff; }
  .btn.ghost { background:transparent; border:1px solid #2a3344; color:var(--ink); }
  table { width:100%; border-collapse:collapse; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid #2a3344; vertical-align:top; }
  .muted { color:var(--muted); font-size:13px; }
  .toolbar { display:flex; gap:8px; padding:16px; align-items:center; flex-wrap:wrap; }
  .toolbar input, .toolbar select { background:#12161e; border:1px solid #2a3344; color:var(--ink); border-radius:8px; padding:8px 10px; }
  .card { margin:16px; padding:16px; background:var(--panel); border-radius:12px; border:1px solid #2a3344; }
  label { display:block; margin:10px 0 4px; color:var(--muted); font-size:13px; }
  .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#243044; font-size:12px; }
  .pill.warn { background:#4a3020; color:#ffb070; }

  .chat-top { display:flex; align-items:center; gap:12px; padding:12px 16px; border-bottom:1px solid #2a3344; }
  .switch { display:inline-flex; align-items:center; gap:8px; cursor:pointer; user-select:none; }
  .switch input { accent-color: var(--accent); width:18px; height:18px; }
  #mic-btn { width:56px; height:56px; border-radius:50%; border:0; background:#2a3344; color:var(--ink);
    font-size:22px; cursor:pointer; flex-shrink:0; display:flex; align-items:center; justify-content:center; }
  #mic-btn.listening { background:#e74c3c; color:#fff; box-shadow:0 0 0 4px rgba(231,76,60,.35); }
  #mic-btn.busy { opacity:.6; }
  #voice-status { font-size:12px; color:var(--muted); min-height:16px; padding:0 16px 4px; }
  .composer { align-items:center; }
</style>
</head>
<body>
<nav>
  <h1 id="agent-title">PC Agent</h1>
  <button class="active" data-view="chat">Chat</button>
  <button data-view="memory">Memory</button>
  <button data-view="voice">Voice</button>
  <button data-view="settings">Settings</button>
  <div style="flex:1"></div>
  <button class="btn danger" onclick="halt()">Kill switch</button>
  <p class="muted">Ctrl+Alt+Esc also stops everything</p>
</nav>
<main>
  <section id="view-chat" class="view active">
    <div class="chat-top">
      <label class="switch" title="Hands-free conversation with Gemini Live">
        <input type="checkbox" id="voice-mode-toggle" onchange="setVoiceMode(this.checked)"/>
        <span>Voice mode: <strong id="voice-mode-label">Off</strong></span>
      </label>
      <span id="voice-engine-pill" class="pill" style="display:none"></span>
      <div style="flex:1"></div>
      <span class="muted" style="font-size:12px">Mic runs in this browser · Ctrl+Alt+Space</span>
    </div>
    <div id="chat-log"></div>
    <div id="voice-status"></div>
    <div class="composer">
      <button id="mic-btn" type="button" title="Click to talk · Hold for push-to-talk"
        aria-label="Talk">🎤</button>
      <input id="chat-input" placeholder="Tell me what to do on this PC…" autofocus/>
      <button onclick="sendChat()">Send</button>
    </div>
    <div id="model-note" class="muted" style="padding:0 16px 8px;font-size:11px;opacity:.7"></div>
  </section>
  <section id="view-memory" class="view">
    <div class="toolbar">
      <input id="mem-q" placeholder="Search memories…" style="flex:1"/>
      <select id="mem-kind">
        <option value="">All kinds</option>
        <option>fact</option><option>preference</option><option>note</option><option>episode</option>
      </select>
      <button class="btn" onclick="loadMemory()">Search</button>
    </div>
    <div style="overflow:auto; padding:0 16px 16px;">
      <table>
        <thead><tr><th>ID</th><th>Kind</th><th>Content</th><th>Tags</th><th></th></tr></thead>
        <tbody id="mem-body"></tbody>
      </table>
    </div>
  </section>
  <section id="view-voice" class="view">
    <div class="card">
      <h2 style="margin-top:0">Voice</h2>
      <p class="muted">Default is Gemini Live native-audio — natural prosody, emotion, breaths, interruptions. TTS is the expressive non-live path. Offline SAPI is robotic and last-resort only. The mic in Chat captures audio in this browser (no extra install).</p>
      <div class="row" style="margin-bottom:12px">
        <label class="switch"><input type="checkbox" id="voice-enabled" onchange="saveVoice()"/> <span>Speak replies aloud: <strong id="voice-enabled-label">On</strong></span></label>
        <button class="btn" type="button" onclick="testSpeak()">Test voice</button>
        <button class="btn ghost" type="button" onclick="setVoiceMode(true)">Start Voice mode</button>
        <button class="btn ghost" type="button" onclick="setVoiceMode(false)">Stop Voice mode</button>
      </div>
      <label>Engine</label>
      <select id="voice-mode" onchange="saveVoice()">
        <option value="live">Live native-audio (default, most human)</option>
        <option value="tts">Expressive Gemini TTS</option>
        <option value="offline">Robotic offline (SAPI / pyttsx3) — last resort</option>
      </select>
      <label>Voice</label>
      <div id="voice-list"></div>
    </div>
  </section>
  <section id="view-settings" class="view">
    <div class="card">
      <h2 style="margin-top:0">Settings</h2>
      <label>Gemini API key</label>
      <div class="row">
        <input id="api-key" type="password" placeholder="AIza…" style="flex:1"/>
        <button class="btn" onclick="saveKey()">Save key</button>
      </div>
      <label>Agent name</label>
      <input id="agent-name" onchange="saveSettings()"/>
      <label>Personality / tone</label>
      <textarea id="personality" rows="3" onchange="saveSettings()"></textarea>
      <label>Chat model</label>
      <input id="model" onchange="saveSettings()"/>
      <div class="row" style="margin-top:12px">
        <label><input type="checkbox" id="proactive" onchange="saveSettings()"/> Proactive mode (disk, downloads, reminders)</label>
      </div>
      <div class="row">
        <label><input type="checkbox" id="screen-aw" onchange="saveSettings()"/> Screen awareness <span class="pill warn">privacy</span> (off by default)</label>
      </div>
      <div class="row">
        <label><input type="checkbox" id="autostart" onchange="saveSettings()"/> Start at login</label>
      </div>
      <div class="row">
        <label><input type="checkbox" id="confirm" onchange="saveSettings()"/> Confirm dangerous actions</label>
      </div>
      <label>Telegram bot token (optional)</label>
      <input id="tg-token" onchange="saveSettings()"/>
      <label>Telegram allowlisted chat id</label>
      <input id="tg-chat" onchange="saveSettings()"/>
    </div>
  </section>
</main>
<script>
const $ = (s) => document.querySelector(s);
document.querySelectorAll('nav button[data-view]').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button[data-view]').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  b.classList.add('active');
  $('#view-' + b.dataset.view).classList.add('active');
  if (b.dataset.view === 'memory') loadMemory();
  if (b.dataset.view === 'voice') loadVoice();
  if (b.dataset.view === 'settings') loadSettings();
});
async function sendChat(){
  const input = $('#chat-input');
  const text = input.value.trim();
  if (!text) return;
  append('user', text); input.value='';
  const el = append('assistant', '…');
  let got = '';
  try {
    const r = await fetch('/api/chat/stream', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = '';
    while (true) {
      const {value, done} = await reader.read(); if (done) break;
      buf += dec.decode(value, {stream:true});
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 2);
        if (!line.startsWith('data: ')) continue;
        const j = JSON.parse(line.slice(6));
        if (j.delta) { got += j.delta; el.textContent = got; $('#chat-log').scrollTop = 1e9; }
        if (j.done && j.model_note) $('#model-note').textContent = j.model_note;
      }
    }
  } catch (e) { if (!got) el.textContent = 'Lost connection to the agent. Try again.'; }
  if (!got) el.textContent = el.textContent === '…' ? '(no reply)' : el.textContent;
}
function append(role, text){
  const d = document.createElement('div');
  d.className = 'msg '+role; d.textContent = text;
  $('#chat-log').appendChild(d); $('#chat-log').scrollTop = 1e9;
  return d;
}
$('#chat-input').addEventListener('keydown', e => { if(e.key==='Enter') sendChat(); });
async function loadMemory(){
  const q = $('#mem-q').value, kind=$('#mem-kind').value;
  const r = await fetch(`/api/memory?q=${encodeURIComponent(q)}&kind=${encodeURIComponent(kind)}`);
  const j = await r.json();
  const body = $('#mem-body'); body.innerHTML='';
  (j.items||[]).forEach(m => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${m.id}</td><td><span class="pill">${m.kind}</span></td><td>${escapeHtml(m.content)}</td><td class="muted">${escapeHtml(m.tags||'')}</td>
      <td><button class="btn ghost" onclick="forgetMem(${m.id})">Forget</button></td>`;
    body.appendChild(tr);
  });
}
async function forgetMem(id){
  if(!confirm('Forget memory #'+id+'?')) return;
  await fetch('/api/memory/'+id, {method:'DELETE'}); loadMemory();
}
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
async function loadVoice(){
  const r = await fetch('/api/voice'); const j = await r.json();
  $('#voice-mode').value = j.mode; $('#voice-enabled').checked = !!j.enabled; $('#voice-enabled-label').textContent = j.enabled ? 'On' : 'Off';
  const list = $('#voice-list'); list.innerHTML='';
  (j.catalog||[]).forEach(v => {
    const row = document.createElement('div'); row.className='row'; row.style.margin='6px 0';
    row.innerHTML = `<label style="display:flex;gap:8px;align-items:center;margin:0">
      <input type="radio" name="voice" value="${v.id}" ${v.id===j.voice_id?'checked':''} onchange="saveVoice()"/> ${escapeHtml(v.label)}</label>
      <button class="btn ghost" onclick="previewVoice('${v.id}')">Preview</button>`;
    list.appendChild(row);
  });
}

/* ---- Voice / mic (browser capture → WebSocket → Live / fallback) ---- */
let voiceWS = null, mediaStream = null, audioCtx = null, processor = null, sourceNode = null;
let talking = false, voiceMode = false, holdTalk = false;
const playQueue = [];
let playing = false, playCtx = null, playTime = 0;

function setVoiceStatus(t){ const el=$('#voice-status'); if(el) el.textContent = t || ''; }
function setVoiceModeLabel(){
  $('#voice-mode-label').textContent = voiceMode ? 'On' : 'Off';
  $('#voice-mode-toggle').checked = voiceMode;
}
async function setVoiceMode(on){
  voiceMode = !!on; setVoiceModeLabel();
  if (voiceMode) { await startTalk({continuous:true}); }
  else { stopTalk(); }
  fetch('/api/voice/mode', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({on: voiceMode})}).catch(()=>{});
}
async function ensureWS(){
  if (voiceWS && voiceWS.readyState === 1) return voiceWS;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  voiceWS = new WebSocket(proto + '://' + location.host + '/ws/voice');
  voiceWS.binaryType = 'arraybuffer';
  await new Promise((res, rej) => {
    voiceWS.onopen = res; voiceWS.onerror = rej;
    setTimeout(() => rej(new Error('ws timeout')), 8000);
  });
  voiceWS.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      try {
        const j = JSON.parse(ev.data);
        if (j.type === 'status') {
          setVoiceStatus(j.state + (j.engine ? ' · ' + j.engine : '') + (j.model ? ' · ' + j.model : ''));
          const pill = $('#voice-engine-pill');
          if (j.engine) { pill.style.display='inline-block'; pill.textContent = j.engine; }
          if (j.state === 'idle' && !voiceMode) stopTalkUi();
        } else if (j.type === 'transcript') {
          append(j.role === 'user' ? 'user' : 'assistant', j.text);
        } else if (j.type === 'error') {
          setVoiceStatus(j.detail || 'Voice error');
        }
      } catch(e){}
      return;
    }
    enqueuePlay(new Int16Array(ev.data));
  };
  voiceWS.onclose = () => { if (voiceMode || talking) setVoiceStatus('Voice connection closed'); };
  return voiceWS;
}
function enqueuePlay(i16){
  playQueue.push(i16);
  if (!playing) drainPlay();
}
async function drainPlay(){
  playing = true;
  try {
    if (!playCtx) playCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate:24000});
    if (playCtx.state === 'suspended') await playCtx.resume();
    while (playQueue.length) {
      const i16 = playQueue.shift();
      const f32 = new Float32Array(i16.length);
      for (let i=0;i<i16.length;i++) f32[i] = i16[i] / 32768;
      const buf = playCtx.createBuffer(1, f32.length, 24000);
      buf.copyToChannel(f32, 0);
      const src = playCtx.createBufferSource();
      src.buffer = buf; src.connect(playCtx.destination);
      const startAt = Math.max(playCtx.currentTime + 0.02, playTime);
      src.start(startAt);
      playTime = startAt + buf.duration;
      await new Promise(r => { src.onended = r; setTimeout(r, buf.duration*1000 + 50); });
    }
  } catch(e) { console.warn('play', e); }
  playing = false;
}
async function startMic(){
  if (mediaStream) return;
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation:true, noiseSuppression:true, channelCount:1 }, video:false });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate:16000});
  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (e) => {
    if (!talking && !voiceMode) return;
    if (!voiceWS || voiceWS.readyState !== 1) return;
    const input = e.inputBuffer.getChannelData(0);
    const i16 = new Int16Array(input.length);
    for (let i=0;i<input.length;i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    voiceWS.send(i16.buffer);
  };
  sourceNode.connect(processor);
  processor.connect(audioCtx.destination);
}
function stopMic(){
  try { processor && processor.disconnect(); sourceNode && sourceNode.disconnect(); } catch(e){}
  processor = sourceNode = null;
  if (audioCtx) { try { audioCtx.close(); } catch(e){} audioCtx = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
}
async function startTalk(opts){
  opts = opts || {};
  try {
    $('#mic-btn').classList.add('busy');
    await ensureWS();
    await startMic();
    voiceWS.send(JSON.stringify({type:'start', continuous: !!opts.continuous || voiceMode}));
    talking = true;
    $('#mic-btn').classList.add('listening');
    $('#mic-btn').classList.remove('busy');
    setVoiceStatus(voiceMode ? 'Voice mode on — listening' : 'Listening… click mic to stop');
  } catch (e) {
    $('#mic-btn').classList.remove('busy','listening');
    setVoiceStatus('Mic error: ' + (e.message || e) + ' — allow microphone for this page.');
    talking = false;
  }
}
function stopTalk(){
  talking = false; holdTalk = false;
  if (voiceWS && voiceWS.readyState === 1) {
    try { voiceWS.send(JSON.stringify({type:'stop'})); } catch(e){}
  }
  stopTalkUi();
  if (!voiceMode) stopMic();
}
function stopTalkUi(){
  $('#mic-btn').classList.remove('listening','busy');
  if (!voiceMode) setVoiceStatus('');
}
const mic = $('#mic-btn');
mic.addEventListener('click', () => {
  if (voiceMode) return; // continuous mode uses the toggle
  if (talking) stopTalk(); else startTalk({continuous:false});
});
mic.addEventListener('mousedown', (e) => { if (voiceMode) return; holdTalk = true; if (!talking) startTalk({continuous:false}); });
mic.addEventListener('mouseup', () => { if (holdTalk) { holdTalk = false; stopTalk(); } });
mic.addEventListener('mouseleave', () => { if (holdTalk) { holdTalk = false; stopTalk(); } });
mic.addEventListener('touchstart', (e) => { e.preventDefault(); if (!voiceMode && !talking) { holdTalk = true; startTalk({continuous:false}); } }, {passive:false});
mic.addEventListener('touchend', () => { if (holdTalk) { holdTalk = false; stopTalk(); } });

async function testSpeak(){
  setVoiceStatus('Testing voice…');
  try {
    const r = await fetch('/api/voice/test', {method:'POST'});
    const j = await r.json();
    setVoiceStatus(j.ok ? 'Test OK — ' + (j.engine || '') : (j.error || 'Test failed'));
    if (j.path) { try { new Audio('/api/file?path='+encodeURIComponent(j.path)).play(); } catch(e){} }
  } catch(e) { setVoiceStatus('Test failed'); }
}
async function saveVoice(){
  const voice = (document.querySelector('input[name=voice]:checked')||{}).value;
  const enabled = $('#voice-enabled').checked;
  $('#voice-enabled-label').textContent = enabled ? 'On' : 'Off';
  await fetch('/api/voice', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({
    mode: $('#voice-mode').value, voice_id: voice, enabled
  })});
}
async function previewVoice(id){
  const r = await fetch('/api/voice/preview', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({voice_id:id})});
  const j = await r.json();
  if(j.path){ new Audio('/api/file?path='+encodeURIComponent(j.path)).play().catch(()=>{}); }
  else alert(j.error || 'Preview failed');
}
async function loadSettings(){
  const r = await fetch('/api/settings'); const j = await r.json();
  $('#agent-name').value = j.agent_name||''; $('#personality').value=j.personality||'';
  $('#model').value=j.model||''; $('#proactive').checked=!!j.proactive;
  $('#screen-aw').checked=!!j.screen_awareness; $('#autostart').checked=!!j.autostart;
  $('#confirm').checked=!!j.confirm_dangerous; $('#tg-token').value=j.telegram_token||'';
  $('#tg-chat').value=(j.telegram_allowlist||[])[0]||'';
  $('#agent-title').textContent = j.agent_name || 'PC Agent';
  $('#api-key').placeholder = j.has_api_key ? '•••• saved' : 'AIza…';
}
async function saveSettings(){
  const chat = $('#tg-chat').value.trim();
  await fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({
    agent_name: $('#agent-name').value, personality: $('#personality').value, model: $('#model').value,
    proactive: $('#proactive').checked, screen_awareness: $('#screen-aw').checked,
    autostart: $('#autostart').checked, confirm_dangerous: $('#confirm').checked,
    telegram_token: $('#tg-token').value, telegram_allowlist: chat ? [chat] : []
  })});
  $('#agent-title').textContent = $('#agent-name').value || 'PC Agent';
}
async function saveKey(){
  await fetch('/api/settings/key', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({key: $('#api-key').value})});
  alert('API key saved to Credential Manager / local store');
}
async function halt(){ await fetch('/api/halt', {method:'POST'}); alert('Agent halted. Resume from tray or restart.'); }
loadSettings();
try { const vp = new URLSearchParams(location.search).get('voice'); if (vp === '1') setVoiceMode(true); } catch(e){}
fetch('/api/status').then(r=>r.json()).then(j=>{ if(j.model_note) $('#model-note').textContent=j.model_note; }).catch(()=>{});
</script>
</body>
</html>
"""


class ChatIn(BaseModel):
    text: str


class KeyIn(BaseModel):
    key: str


class VoiceIn(BaseModel):
    mode: str | None = None
    voice_id: str | None = None
    enabled: bool | None = None


class PreviewIn(BaseModel):
    voice_id: str | None = None


def create_app(
    chat_fn: Callable[[str], str],
    memory: "MemoryStore",
    voice: "VoiceEngine | None",
    get_settings: Callable[[], dict],
    update_settings: Callable[[dict], None],
    set_key: Callable[[str], None],
    halt_fn: Callable[[], None],
    resume_fn: Callable[[], None] | None = None,
    status_fn: Callable[[], str] | None = None,
    stream_fn: Callable[[str], Any] | None = None,
    voice_hub: Any | None = None,
) -> FastAPI:
    app = FastAPI(title="PC Agent")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.post("/api/chat")
    def api_chat(body: ChatIn) -> dict:
        try:
            reply = chat_fn(body.text)
            return {"reply": reply, "model_note": status_fn() if status_fn else ""}
        except Exception:
            logger.exception("chat failed")
            return {"reply": "Something went wrong on my side. Try again in a moment.",
                    "model_note": status_fn() if status_fn else ""}

    @app.post("/api/chat/stream")
    def api_chat_stream(body: ChatIn):
        def gen():
            try:
                source = stream_fn(body.text) if stream_fn else iter([chat_fn(body.text)])
                for delta in source:
                    yield "data: " + _json.dumps({"delta": delta}) + "\n\n"
            except Exception:
                logger.exception("stream failed")
                yield "data: " + _json.dumps({"delta": "Something went wrong on my side. Try again in a moment."}) + "\n\n"
            yield "data: " + _json.dumps({"done": True, "model_note": status_fn() if status_fn else ""}) + "\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


    @app.websocket("/ws/voice")
    async def ws_voice(ws):
        await ws.accept()
        if voice_hub is None:
            await ws.send_json({"type": "error", "detail": "Voice not ready — add your Gemini API key in Settings."})
            await ws.close()
            return
        await voice_hub.handle(ws)

    @app.post("/api/voice/mode")
    def api_voice_mode(body: dict) -> dict:
        if voice_hub is None:
            return {"ok": False, "error": "voice not ready"}
        on = bool(body.get("on"))
        voice_hub.set_mode(on)
        return {"ok": True, "on": on}

    @app.post("/api/voice/test")
    def api_voice_test() -> dict:
        if voice is None:
            return {"ok": False, "error": "voice engine unavailable"}
        return voice.preview()

    @app.get("/api/status")
    def api_status() -> dict:
        return {"model_note": status_fn() if status_fn else ""}

    @app.get("/api/memory")
    def api_memory(q: str = "", kind: str = "") -> dict:
        kinds = [kind] if kind else None
        if q:
            items = memory.recall(q, limit=50, kinds=kinds, use_embeddings=True)
        else:
            items = memory.list_memories(limit=100, kinds=kinds)
        return {"items": [i.to_dict() for i in items]}

    @app.delete("/api/memory/{memory_id}")
    def api_forget(memory_id: int) -> dict:
        n = memory.forget(memory_id=memory_id)
        return {"ok": True, "deleted": n}

    @app.get("/api/voice")
    def api_voice() -> dict:
        from pc_agent.config import VOICE_CATALOG, load_config

        cfg = load_config()
        return {
            "mode": cfg.get("voice_mode", "live"),
            "voice_id": cfg.get("voice_id"),
            "enabled": cfg.get("voice_enabled", True),
            "catalog": VOICE_CATALOG,
            "offline_label": "Robotic offline (SAPI / pyttsx3) — last-resort fallback",
        }

    @app.post("/api/voice")
    def api_voice_set(body: VoiceIn) -> dict:
        from pc_agent.config import load_config, save_config

        cfg = load_config()
        if body.mode is not None:
            cfg["voice_mode"] = body.mode
        if body.voice_id is not None:
            cfg["voice_id"] = body.voice_id
        if body.enabled is not None:
            cfg["voice_enabled"] = body.enabled
        save_config(cfg)
        return {"ok": True}

    @app.post("/api/voice/preview")
    def api_voice_preview(body: PreviewIn) -> dict:
        if voice is None:
            return {"ok": False, "error": "voice engine unavailable"}
        return voice.preview(body.voice_id)

    @app.get("/api/file")
    def api_file(path: str):
        p = Path(path)
        # Only allow files under data_dir
        from pc_agent.config import data_dir

        try:
            p.resolve().relative_to(data_dir().resolve())
        except Exception:
            raise HTTPException(403, "path not allowed")
        if not p.is_file():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/settings")
    def api_settings() -> dict:
        s = get_settings()
        from pc_agent.config import get_api_key

        s = dict(s)
        s["has_api_key"] = bool(get_api_key())
        return s

    @app.post("/api/settings")
    def api_settings_set(body: dict) -> dict:
        update_settings(body)
        return {"ok": True}

    @app.post("/api/settings/key")
    def api_key(body: KeyIn) -> dict:
        set_key(body.key)
        return {"ok": True}

    @app.post("/api/halt")
    def api_halt() -> dict:
        halt_fn()
        return {"ok": True}

    @app.post("/api/resume")
    def api_resume() -> dict:
        if resume_fn:
            resume_fn()
        return {"ok": True}

    return app


def run_server(app: FastAPI, host: str, port: int) -> threading.Thread:
    import uvicorn

    def _run() -> None:
        uvicorn.run(app, host=host, port=port, log_level="warning")

    t = threading.Thread(target=_run, name="pc-agent-ui", daemon=True)
    t.start()
    return t
