# PC Agent

Always-on Windows desktop agent with a Gemini brain. Controls your whole PC — shell, files, apps, mouse/keyboard, windows, clipboard, screenshots + vision, browser automation, scheduling — and talks back with a **human** voice.

Built to be better than a chatbox: long-term memory, proactive notices, multi-step planning with screenshot self-check, reusable skills, and plugins.

## Download

After CI publishes a release:

- **Installer:** `https://github.com/lhymite36-bot/pc-agent/releases/latest/download/PCAgent-Setup.exe`
- **Repo:** https://github.com/lhymite36-bot/pc-agent

## Setup (Windows)

1. Download and run `PCAgent-Setup.exe` (or run from source — below).
2. On first launch the chat UI opens at http://127.0.0.1:8765/
3. Open **Settings** → paste your **Gemini API key** → Save. It is stored in **Windows Credential Manager** (via `keyring`).
4. Pick a voice under **Voice** (default: **Sulafat — Warm**, Live native-audio).
5. Leave it running in the tray. Enable **Start at login** for 24/7.

### Kill switch

**Ctrl+Alt+Esc** immediately halts all actions. Resume from the tray menu or the UI.

Dangerous actions (delete, install, send message/email, payments, registry, shutdown, Chrome profile) show a confirmation before running.

## Features

### Brain
- Gemini function calling; model is configurable and discovered at runtime (default `gemini-3.8-flash`).

### Automatic model failover
A ranked list of good text models is built from `models.list` (cached ~3 h): newest flash first, then pro, then other flash — no lite/TTS/embedding/image/Veo/Live. On 503/429/404/500 it retries once, then silently moves to the next model, sticks with the working one for 15 min, then tries the preferred model again. A 404 on the configured model saves the replacement to config. The UI shows a tiny “Using <model>” note; if everything fails you get a friendly message, never a raw error. Vision, planner, screen awareness, summaries and embeddings share the same failover; TTS and Live voice walk their own expressive model lists, then SAPI.

### Offline STT + quota (1.0.7)
- **Local STT by default** (faster-whisper `base` int8 on CPU; model downloads once into the data dir). No free-tier quota spent on transcription.
- 429s fail over across flash models with `retryDelay` cooldowns (persisted). Live / TTS / chat share the same health store.
- Clear one-line notice when free quota is gone (local reset time + billing hint); replies fall back to Windows SAPI.
- Memory extraction stays heuristic; embeddings/summaries are batched rarely and never call Gemini for summaries.
- Installer upgrades while the app is running: AppMutex, CloseApplications, and `taskkill` of PCAgent/AtlasCLI before copy.

### Terminal mode (1.0.6)
- **Atlas (Terminal)** Start Menu shortcut → `AtlasCLI.exe` (console), or `PCAgent.exe --cli`.
- Streaming text chat in the terminal (rich). Same brain, tools, memory, failover, safety.
- Native mic/speakers via **sounddevice** (PortAudio bundled). Enter toggles push-to-talk; `/voice` for hands-free Gemini Live; falls back to STT→chat→TTS→SAPI.
- Commands: `/voice` `/text` `/mic` `/model` `/memory` `/quit`. Confirms are `y/N` in the terminal; Ctrl+Alt+Esc still kills actions.
- Safe alongside the tray/web app: SQLite **WAL** mode. CI runs `AtlasCLI.exe --selftest` on Windows.

### Reliability (1.0.5)
- Hard timeouts: ~10 s to first token, ~60 s total stream — then cancel and fail over.
- Network errors (`RemoteProtocolError`, timeouts, disconnects) cool the model and fail over.
- SDK HTTP retries disabled (`HttpRetryOptions(attempts=1)`); we own failover timing.
- Models ranked by measured first-token latency (fast healthy flash first). Slow models (e.g. a hung 3.8-flash) get a cooldown.
- Chat shows **thinking…** until the first token; always ends with a reply or a friendly error.

### Voice UI (1.0.4)
- Large **Talk** mic button next to the chat input (click-to-talk and hold-to-talk).
- **Voice mode: On/Off** switch on the Chat page for hands-free Gemini Live (falls back to STT→chat→TTS).
- Browser `getUserMedia` mic → WebSocket → agent; playback in the browser (no extra Python audio deps required).
- Voice tab: on/off + **Test voice**. Tray **Voice mode** item and **Ctrl+Alt+Space** hotkey.
- PyInstaller collects `pc_agent.core.voice`, `live_session`, `voice_hub`, `hotkeys`.

### Speed (1.0.3)
- Replies stream into the chat window token by token, and speech starts sentence by sentence.
- Normal chat uses minimal thinking; the planner uses high thinking.
- Failover: a 404 marks the model dead, a 429 cools it down for its `retryDelay`, a 503 gets one 300 ms retry. Dead and cooling models are remembered across restarts (`model_health.json`). `models.list` is refreshed in the background only (`models_cache.json`).
- Memory extraction, embeddings and summarization run in a background thread after the reply. Prompt recall is local FTS.
- Compact system prompt, capped history window, trimmed tool outputs.
- Per-stage timings go to `%LOCALAPPDATA%\pc-agent\agent.log` (`pc_agent.timing`).

### Voice (realistic / human)
| Priority | Engine | Notes |
|---|---|---|
| 1 (default) | **Gemini Live** `gemini-3.8-live` | Native audio — natural prosody, emotion, breaths, barge-in. Streams PCM for low latency. |
| 2 | **Gemini TTS** `gemini-3.8-flash-tts` | Expressive non-live path with a conversational style prompt. Not the flat lite models. |
| 3 | **Offline SAPI / pyttsx3** | Robotic last-resort fallback — labeled as such in Settings. |

Warm default voice: **Sulafat**. Voice picker with previews. Style: *conversational, casual, contractions* — not reading a script.

### Memory (first-class)
- SQLite long-term store with **remember / recall / forget** tools the model calls itself.
- Automatic extraction of durable facts & preferences from chat.
- Recall into every prompt (FTS5 keyword search + Gemini embeddings when available).
- Conversation history across restarts with rolling summarization.
- Task / episode log.
- **Memory** view in the UI: list, search, delete.

### Proactive mode
Notices low disk, stuck downloads, due reminders — speaks up / logs an episode. Toggle in Settings.

### Screen awareness (privacy)
Optional periodic screen understanding so “what's this error?” is instant. **Off by default.**

### Planner
`run_plan` — plan → execute tools → verify with screenshot → retry once.

### Browser
Playwright tools. Using your Chrome profile requires an explicit confirmation.

### Skills & plugins
- Save repeated workflows as named JSON skills (`save_skill` / `run_skill`).
- Drop custom Python tools into `%LOCALAPPDATA%\pc-agent\plugins\` (`DECLARATIONS` + `HANDLERS`).

### Personality
Name + tone in Settings. Default agent name **Atlas**, witty/warm.

### Telegram (optional)
Bot token + allowlisted chat id. Ignores everyone else.

## Run from source

```bash
pip install -e ".[dev]"
python -m pc_agent
```

24/7 supervisor (restart on crash):

```bash
python -m pc_agent --watchdog
```

## Tests

```bash
pytest -q
```

GUI / OS automation is mocked or skipped on Linux CI. Windows build is produced by GitHub Actions (`windows-latest`) with PyInstaller + Inno Setup.

## Safety

- Confirmation gate for destructive / external actions.
- Global kill-switch hotkey.
- Append-only action log: `%LOCALAPPDATA%\pc-agent\actions.log`
- Screen awareness and Chrome-profile browser use are opt-in.

## Limitations

- Full desktop control (mouse/keyboard/windows) requires a real Windows session (not a headless RDP lock screen).
- Gemini Live / TTS need network + quota on your API key.
- Playwright is optional; install with `pip install playwright && playwright install chromium` for browser tools.
- Offline voice is intentionally robotic.
- Confirmation prompts in fully headless mode deny by default (safe).

## License

MIT — for Pedro's personal use.
