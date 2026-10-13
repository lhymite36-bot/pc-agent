"""Atlas terminal mode — text + native mic voice (no browser)."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from typing import Iterator

logger = logging.getLogger("pc_agent.cli")
timing = logging.getLogger("pc_agent.timing")


def selftest() -> int:
    """CI / frozen-exe smoke test. Exit 0 on success."""
    errors: list[str] = []
    print("AtlasCLI --selftest")

    # Config
    try:
        from pc_agent.config import VERSION, load_config, data_dir, DEFAULT_MODEL

        cfg = load_config()
        print(f"  version={VERSION} model_default={DEFAULT_MODEL} data_dir={data_dir()}")
        print(f"  config_keys={sorted(cfg.keys())[:6]}...")
    except Exception as exc:
        errors.append(f"config: {exc}")
        print(f"  FAIL config: {exc}")

    # Core imports
    try:
        from pc_agent.core.brain import Brain  # noqa: F401
        from pc_agent.core.memory import MemoryStore
        from pc_agent.core.models import ModelRouter  # noqa: F401
        from pc_agent.core.voice import VoiceEngine  # noqa: F401
        from pc_agent.core.live_session import LiveVoiceSession  # noqa: F401
        from pc_agent.core.terminal_audio import list_devices, sounddevice_available
        from pc_agent.core import local_stt

        print("  imports=ok")
        ok_stt, stt_detail = local_stt.available()
        print(f"  faster_whisper={'ok' if ok_stt else stt_detail}")
    except Exception as exc:
        errors.append(f"imports: {exc}")
        print(f"  FAIL imports: {exc}")
        _print_result(errors)
        return 1

    # sounddevice / PortAudio
    ok, detail = sounddevice_available()
    if not ok:
        errors.append(f"sounddevice: {detail}")
        print(f"  FAIL sounddevice: {detail}")
    else:
        try:
            devices = list_devices()
            print(f"  sounddevice=ok devices={len(devices)}")
            for d in devices[:5]:
                print(f"    [{d['index']}] in={d['max_input_channels']} out={d['max_output_channels']} {d['name']}")
        except Exception as exc:
            errors.append(f"query_devices: {exc}")
            print(f"  FAIL query_devices: {exc}")

    # Memory WAL
    try:
        import tempfile
        from pathlib import Path

        db = Path(tempfile.mkdtemp()) / "t.db"
        store = MemoryStore(path=db)
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        store.remember("selftest", kind="note")
        store.close()
        print(f"  sqlite_journal={mode}")
        if str(mode).lower() != "wal":
            errors.append(f"expected WAL, got {mode}")
    except Exception as exc:
        errors.append(f"memory: {exc}")
        print(f"  FAIL memory: {exc}")

    return _print_result(errors)


def _print_result(errors: list[str]) -> int:
    if errors:
        print("SELFTEST FAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("SELFTEST OK")
    return 0


class TerminalApp:
    def __init__(self, agent) -> None:
        from pc_agent.config import load_config

        self.agent = agent
        self.cfg = load_config()
        self.mode = "text"  # text | ptt | voice
        self.mic_device: int | None = self.cfg.get("cli_mic_device")
        self.spk_device: int | None = self.cfg.get("cli_spk_device")
        self._live = None
        self._speaker = None
        self._listening = False
        self._console = None
        try:
            from rich.console import Console

            self._console = Console()
        except Exception:
            pass

    # --- output helpers ---

    def print(self, *args, style: str | None = None, **kwargs) -> None:
        if self._console is not None:
            self._console.print(*args, style=style, **kwargs)
        else:
            print(*args, **kwargs)

    def status(self, msg: str) -> None:
        self.print(f"[dim]{msg}[/dim]" if self._console else msg)

    def banner(self) -> None:
        from pc_agent import __version__
        from pc_agent.config import load_config

        cfg = load_config()
        name = cfg.get("agent_name") or "Atlas"
        self.print(
            f"\n[bold cyan]{name}[/bold cyan] terminal v{__version__}  "
            f"[dim]model={cfg.get('model')} · /help for commands[/dim]\n"
            if self._console
            else f"\n{name} terminal v{__version__}  model={cfg.get('model')} · /help\n"
        )

    # --- confirm in terminal ---

    def install_confirm(self) -> None:
        from pc_agent.core import safety

        def _confirm(category: str, description: str) -> bool:
            self.print(f"\n[yellow]Confirm {category}[/yellow]: {description}", style=None)
            try:
                ans = input("Allow? [y/N] ").strip().lower()
            except EOFError:
                return False
            return ans in {"y", "yes"}

        safety.set_confirm_callback(_confirm)
        safety.start_kill_switch()

    # --- commands ---

    def handle_command(self, line: str) -> bool:
        """Return True if handled as a command (including quit)."""
        parts = line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"/quit", "/exit", "/q"}:
            self.shutdown()
            return True
        if cmd in {"/help", "/?"}:
            self.print(
                "Commands:\n"
                "  /voice          Hands-free Gemini Live (mic + speakers)\n"
                "  /text           Text mode (default)\n"
                "  Enter (empty)   Toggle push-to-talk in text mode\n"
                "  /mic            List audio devices\n"
                "  /mic in|out N   Select device\n"
                "  /model [NAME]   Show / prefer Gemini model\n"
                "  /keys           List API keys (rotation)\n"
                "  /keys add KEY   Add another free Gemini API key\n"
                "  /keys rotate    Switch to the next key\n"
                "  /memory [q]     Search / list memories\n"
                "  /ollama         Check local Ollama fallback\n"
                "  /quit           Leave\n"
                "Offline (no quota): open chrome/notepad/downloads, volume, screenshot, lock, time…\n"
            )
            return True
        if cmd == "/voice":
            self.start_voice_mode()
            return True
        if cmd == "/text":
            self.stop_voice_mode()
            self.mode = "text"
            self.status("Text mode.")
            return True
        if cmd == "/mic":
            self.cmd_mic(arg)
            return True
        if cmd == "/model":
            self.cmd_model(arg)
            return True
        if cmd == "/memory":
            self.cmd_memory(arg)
            return True
        if cmd == "/keys":
            self.cmd_keys(arg)
            return True
        if cmd == "/ollama":
            self.cmd_ollama()
            return True
        if cmd.startswith("/"):
            self.print(f"Unknown command: {cmd}  (try /help)")
            return True
        return False

    def cmd_mic(self, arg: str) -> None:
        from pc_agent.core.terminal_audio import default_devices, list_devices, sounddevice_available
        from pc_agent.config import load_config, save_config

        ok, detail = sounddevice_available()
        if not ok:
            self.print(f"sounddevice unavailable: {detail}")
            return
        if not arg:
            defs = default_devices()
            self.print(f"Default in={defs['input']} out={defs['output']}")
            self.print(f"Selected in={self.mic_device} out={self.spk_device}")
            for d in list_devices():
                flags = []
                if d["max_input_channels"]:
                    flags.append("IN")
                if d["max_output_channels"]:
                    flags.append("OUT")
                self.print(f"  [{d['index']}] {'/'.join(flags):7} {d['name']}")
            return
        toks = arg.split()
        if len(toks) == 2 and toks[0] in {"in", "out"} and toks[1].isdigit():
            n = int(toks[1])
            cfg = load_config()
            if toks[0] == "in":
                self.mic_device = n
                cfg["cli_mic_device"] = n
            else:
                self.spk_device = n
                cfg["cli_spk_device"] = n
            save_config(cfg)
            self.status(f"Mic {toks[0]} device = {n}")
            return
        self.print("Usage: /mic | /mic in N | /mic out N")

    def cmd_model(self, arg: str) -> None:
        from pc_agent.config import load_config, save_config

        cfg = load_config()
        if not arg:
            note = self.agent.brain.model_note if self.agent.brain else ""
            self.print(f"Configured: {cfg.get('model')}  Live: {note}")
            if self.agent.brain:
                ranked = self.agent.brain.list_models()[:12]
                self.print("Ranked:", ", ".join(ranked) if ranked else "(cache empty)")
            return
        cfg["model"] = arg.strip()
        save_config(cfg)
        if self.agent.brain:
            self.agent.brain.pick_model(arg.strip())
        self.status(f"Preferred model set to {arg.strip()}")

    def cmd_memory(self, arg: str) -> None:
        mem = self.agent.memory
        items = mem.recall(arg, limit=15, use_embeddings=False) if arg else mem.list_memories(limit=15)
        if not items:
            self.print("(no memories)")
            return
        for m in items:
            self.print(f"  #{m.id} [{m.kind}] {m.content}")

    # --- chat ---

    def chat_line(self, text: str) -> None:
        self.print(f"[bold green]You[/bold green] {text}" if self._console else f"You: {text}")
        name = (self.cfg.get("agent_name") or "Atlas")
        prefix = f"[bold cyan]{name}[/bold cyan] " if self._console else f"{name}: "

        # Fast offline path — $0, instant
        from pc_agent.core.offline_router import try_handle
        offline = try_handle(text)
        if offline is not None:
            if self._console:
                self._console.print(prefix, end="")
                self._console.print(offline)
            else:
                print(prefix + offline)
            self._speak_free(offline)
            self.agent.memory.add_message("user", text)
            self.agent.memory.add_message("assistant", offline)
            return

        if self._console:
            self._console.print(prefix, end="")
        else:
            print(prefix, end="", flush=True)
        reply_parts: list[str] = []
        try:
            for delta in self.agent.chat_stream(text):
                reply_parts.append(delta)
                if self._console:
                    self._console.print(delta, end="", highlight=False)
                else:
                    print(delta, end="", flush=True)
        except Exception:
            logger.exception("chat failed")
            reply_parts = []
        reply = "".join(reply_parts)
        print()

        # If Gemini is out, try Ollama, then offline apology with tips
        from pc_agent.core.quota import is_quota_exhausted, format_quota_notice, model_reset_map
        if (not reply) or is_quota_exhausted(reply) or reply.startswith("All the Gemini"):
            local = self._ollama_chat(text)
            if local:
                self.print(f"\n[dim](local Ollama)[/dim] {local}" if self._console else f"\n(local Ollama) {local}")
                self._speak_free(local)
                return
            if is_quota_exhausted(reply) or (reply or "").startswith("All the Gemini"):
                # Enrich notice with per-model resets; strip any old billing wording
                health = getattr(getattr(self.agent, "brain", None), "health", None)
                resets = model_reset_map(health) if health else None
                notice = format_quota_notice(reply or "429", model_resets=resets)
                if reply and reply != notice:
                    # Replace the streamed quota line visually
                    pass
                self.print(f"[dim]{notice}[/dim]" if self._console else notice)
                tip = try_handle("what time is it")  # sanity
                self.status("Tip: try “open chrome”, “open downloads”, “screenshot”, “volume mute”.")
                self._speak_free("I'm on free offline mode for a bit. Try opening an app or asking for the time.")
                return
        if reply:
            self._speak_free(reply)

    # --- push-to-talk ---

    def toggle_talk(self) -> None:
        if self.mode == "voice":
            self.status("Already in /voice mode. /text to leave.")
            return
        if self._listening:
            self._listening = False
            return
        self._listening = True
        self.status("Listening… speak, then pause (or press Enter).")
        threading.Thread(target=self._ptt_capture, name="cli-ptt", daemon=True).start()

    def _ptt_capture(self) -> None:
        from pc_agent.core.terminal_audio import MicRecorder, sounddevice_available

        ok, detail = sounddevice_available()
        if not ok:
            self.print(f"Mic unavailable: {detail}")
            self._listening = False
            return
        rec = MicRecorder(device=self.mic_device)
        try:
            pcm = rec.read_utterance()
        except Exception as exc:
            self.print(f"Mic error: {exc}")
            self._listening = False
            return
        self._listening = False
        if len(pcm) < 3200:
            self.status("Didn't catch that.")
            return
        text = self._stt(pcm)
        if not text:
            self.status("Couldn't transcribe.")
            return
        self.chat_line(text)
        # Speak reply if voice_enabled
        from pc_agent.config import load_config

        if load_config().get("voice_enabled", True):
            # Last assistant message — re-speak short form via TTS to speakers
            pass  # chat_stream already queues speech worker; ensure speaker path
            self._ensure_browserless_tts()

    def _ensure_browserless_tts(self) -> None:
        """Point voice engine playback at sounddevice speakers."""
        from pc_agent.core.terminal_audio import Speaker

        if self._speaker is None:
            self._speaker = Speaker(device=self.spk_device)
        eng = self.agent.voice
        speaker = self._speaker
        # Monkey-patch play to use our speaker (CLI has no winsound preference)
        original = eng._play_pcm_chunk

        def _play(pcm: bytes, rate: int = 24000) -> None:
            try:
                if rate != speaker.rate:
                    speaker.rate = rate
                    speaker.close()
                speaker.write(pcm)
            except Exception:
                original(pcm, rate=rate)

        eng._play_pcm_chunk = _play  # type: ignore[method-assign]

    def _stt(self, pcm: bytes) -> str:
        """Local offline STT first (no quota). Cloud STT only as last resort with failover."""
        # 1) Offline faster-whisper
        try:
            from pc_agent.core.local_stt import available, transcribe

            ok, detail = available()
            if ok:
                self.status("Transcribing offline…")
                text = transcribe(pcm, on_progress=self.status)
                if text:
                    return text
            else:
                logger.info("local STT unavailable: %s", detail)
        except Exception as exc:
            logger.warning("local STT failed: %s", exc)
            self.status(f"Offline STT issue: {exc}")

        # 2) Cloud STT with shared model failover (only if local failed)
        return self._stt_cloud(pcm)

    def _stt_cloud(self, pcm: bytes) -> str:
        from pc_agent.config import FALLBACK_MODELS, get_api_key, load_config
        from pc_agent.core.models import AllModelsFailed, ModelRouter, is_retryable
        from pc_agent.core.quota import format_quota_notice, is_quota_exhausted

        key = get_api_key()
        if not key or not self.agent.brain:
            return ""
        import io, wave
        from google.genai import types

        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
            wf.writeframes(pcm)
        wav = bio.getvalue()
        brain = self.agent.brain
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="Transcribe this speech exactly. Reply with only the transcript."),
                    types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                ],
            )
        ]

        def _call(model):
            return brain.client.models.generate_content(model=model, contents=contents)

        try:
            # Reuse text router so 429s cool models with retryDelay
            resp = brain.router.call(_call)
            return (resp.text or "").strip()
        except AllModelsFailed as exc:
            if is_quota_exhausted(exc):
                notice = format_quota_notice(exc)
                self.print(notice)
                self._force_offline_voice()
            else:
                self.status("Cloud STT unavailable on every model.")
            return ""
        except Exception as exc:
            logger.warning("cloud STT failed: %s", exc)
            if is_quota_exhausted(exc):
                self.print(format_quota_notice(exc))
                self._force_offline_voice()
            return ""

    def _force_offline_voice(self) -> None:
        from pc_agent.config import load_config, save_config
        cfg = load_config()
        if cfg.get("voice_mode") != "offline":
            cfg["voice_mode"] = "offline"
            save_config(cfg)
            self.status("Switched voice to offline SAPI until quota resets.")

    # --- hands-free Live ---

    def start_voice_mode(self) -> None:
        from pc_agent.config import get_api_key
        from pc_agent.core.live_session import LiveVoiceSession
        from pc_agent.core.terminal_audio import MicRecorder, Speaker, sounddevice_available

        ok, detail = sounddevice_available()
        if not ok:
            self.print(f"Mic/speaker unavailable: {detail}")
            return
        key = get_api_key()
        if not key:
            self.print("Add your Gemini API key first (run the tray app Settings once, or set it via keyring).")
            return
        self.stop_voice_mode()
        self.mode = "voice"
        self._ensure_browserless_tts()
        speaker = self._speaker or Speaker(device=self.spk_device)
        self._speaker = speaker

        def on_pcm(pcm: bytes) -> None:
            speaker.write(pcm)

        def on_event(ev: dict) -> None:
            typ = ev.get("type")
            if typ == "transcript":
                role = ev.get("role")
                text = ev.get("text") or ""
                if role == "user":
                    self.print(f"[green]You[/green] {text}" if self._console else f"You: {text}")
                else:
                    self.print(f"[cyan]Atlas[/cyan] {text}" if self._console else f"Atlas: {text}")
            elif typ == "status":
                self.status(f"voice: {ev.get('state')} {ev.get('engine', '')} {ev.get('model', '')}".strip())

        session = LiveVoiceSession(
            api_key=key,
            on_pcm_out=on_pcm,
            on_event=on_event,
            chat_fn=self.agent.chat,
            voice_engine=self.agent.voice,
        )
        self._live = session
        session.start(continuous=True)

        # Feed mic into Live
        def _feed():
            rec = MicRecorder(device=self.mic_device, on_chunk=session.push_pcm)
            try:
                rec.start()
                while self.mode == "voice" and session.active:
                    time.sleep(0.2)
            finally:
                rec.stop()

        threading.Thread(target=_feed, name="cli-live-mic", daemon=True).start()
        self.status("Voice mode ON — speak freely. /text to stop.")

    def stop_voice_mode(self) -> None:
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
        self.mode = "text"


    def _speak_free(self, text: str) -> None:
        """Default CLI voice: edge-tts or SAPI — not Gemini (saves quota, starts fast)."""
        from pc_agent.config import load_config
        if not load_config().get("voice_enabled", True):
            return
        # Shorten for speech
        spoken = text.strip()
        if len(spoken) > 400:
            spoken = spoken[:400].rsplit(" ", 1)[0] + "…"
        def _run():
            try:
                from pc_agent.core.free_tts import speak_free
                from pc_agent.core.terminal_audio import Speaker
                if self._speaker is None:
                    self._speaker = Speaker(device=self.spk_device)
                spk = self._speaker
                def on_chunk(pcm: bytes):
                    try:
                        spk.write(pcm)
                    except Exception:
                        pass
                # Prefer SAPI for sub-second start; edge-tts if configured
                mode = load_config().get("cli_tts", "sapi")  # sapi | edge
                if mode == "edge":
                    speak_free(spoken, on_chunk=None)
                else:
                    from pc_agent.core.free_tts import _speak_sapi
                    _speak_sapi(spoken)
            except Exception:
                logger.exception("free tts failed")
        threading.Thread(target=_run, name="cli-tts", daemon=True).start()

    def _ollama_chat(self, text: str) -> str:
        try:
            from pc_agent.core import ollama as ol
            ok, models = ol.available()
            if not ok:
                return ""
            self.status("Gemini quota empty — trying local Ollama…")
            msgs = [
                {"role": "system", "content": "You are Atlas, a concise PC helper. Be brief."},
                {"role": "user", "content": text},
            ]
            return ol.chat(msgs)
        except Exception as exc:
            logger.info("ollama fallback skipped: %s", exc)
            return ""

    def cmd_keys(self, arg: str) -> None:
        from pc_agent.core import keys as keystore
        arg = (arg or "").strip()
        if not arg:
            rows = keystore.list_status()
            if not rows:
                self.print("No API keys yet. Add one with: /keys add AIza…")
                return
            for r in rows:
                flag = "*" if r["active"] else " "
                cool = f"  cooling until {r['reset_local']} local" if r["cooling"] else ""
                self.print(f" {flag} {r['id']:10} {r['hint']}{cool}")
            return
        if arg.startswith("add "):
            secret = arg[4:].strip()
            info = keystore.add_key(secret)
            self.print(f"Added {info['id']} ({info['hint']})")
            # Re-init brain with new active key
            try:
                self.agent._set_key(keystore.active_key() or secret)
            except Exception:
                pass
            return
        if arg == "rotate":
            k = keystore.rotate()
            self.print("Rotated." if k else "No other key available.")
            if k:
                try:
                    self.agent._set_key(k)
                except Exception:
                    pass
            return
        self.print("Usage: /keys | /keys add AIza… | /keys rotate")

    def cmd_ollama(self) -> None:
        from pc_agent.core import ollama as ol
        ok, models = ol.available()
        if not ok:
            self.print("Ollama not detected at localhost:11434.")
            self.print("Optional free local brain:  ollama pull llama3.2:3b")
            return
        pick = ol.pick_model(models)
        self.print(f"Ollama OK · {len(models)} model(s) · will use {pick}")

    def shutdown(self) -> None:
        self.stop_voice_mode()
        if self._speaker:
            self._speaker.close()
        self.print("Bye.")
        raise SystemExit(0)

    # --- main loop ---

    def run(self) -> None:
        self.install_confirm()
        self._ensure_browserless_tts()
        self.banner()
        while True:
            try:
                prompt = "you› " if self.mode == "text" else "voice› "
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                self.shutdown()
                return
            if line is None:
                continue
            stripped = line.strip()
            if stripped.startswith("/"):
                try:
                    self.handle_command(stripped)
                except SystemExit:
                    raise
                except Exception:
                    logger.exception("command failed")
                continue
            if stripped == "":
                # Enter toggles push-to-talk
                self.toggle_talk()
                continue
            if self.mode == "voice":
                # Typed text while in voice mode still chats
                self.chat_line(stripped)
            else:
                self.chat_line(stripped)


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="AtlasCLI", description="Atlas terminal agent")
    parser.add_argument("--selftest", action="store_true", help="Smoke-test imports/audio/config and exit")
    parser.add_argument("--version", action="store_true")
    args, _unknown = parser.parse_known_args(argv)

    if args.version:
        from pc_agent import __version__

        print(__version__)
        return 0
    if args.selftest:
        return selftest()

    # Full agent without web UI (safe alongside tray if already on :8765)
    from pc_agent.app import Agent
    from pc_agent.config import get_api_key

    _setup_cli_logging()
    agent = Agent()
    # Bootstrap brain/tools/scheduler but do NOT bind the web port
    agent.bootstrap()
    if not get_api_key():
        print("No Gemini API key yet.")
        print("Open the desktop app once and save your key in Settings, or set it via keyring.")
        print("Continuing — /model and chat will ask again after you add a key.")
    app = TerminalApp(agent)
    try:
        app.run()
    except SystemExit:
        agent.memory.close()
        return 0
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_cli(argv))


if __name__ == "__main__":
    main()
