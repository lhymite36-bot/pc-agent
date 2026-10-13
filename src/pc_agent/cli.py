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

        print("  imports=ok")
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
                "  /mic in N       Select input device N\n"
                "  /mic out N      Select output device N\n"
                "  /model          Show current model\n"
                "  /model NAME     Prefer model NAME (saved)\n"
                "  /memory [q]     Search / list memories\n"
                "  /quit /exit     Leave the terminal\n"
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
        if self._console:
            self._console.print(prefix, end="")
        else:
            print(prefix, end="", flush=True)
        try:
            for delta in self.agent.chat_stream(text):
                if self._console:
                    self._console.print(delta, end="", highlight=False)
                else:
                    print(delta, end="", flush=True)
        except Exception:
            logger.exception("chat failed")
            msg = "Something went wrong. Try again."
            self.print(msg)
            return
        print()

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
        self.status("Transcribing…")
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
        key = __import__("pc_agent.config", fromlist=["get_api_key"]).get_api_key()
        if not key:
            return ""
        try:
            from google import genai
            from google.genai import types
            from pc_agent.config import FALLBACK_MODELS, load_config
            import io
            import wave

            client = genai.Client(api_key=key)
            model = load_config().get("model") or FALLBACK_MODELS[0]
            bio = io.BytesIO()
            with wave.open(bio, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm)
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text="Transcribe this speech exactly. Reply with only the transcript."
                            ),
                            types.Part.from_bytes(data=bio.getvalue(), mime_type="audio/wav"),
                        ],
                    )
                ],
            )
            return (resp.text or "").strip()
        except Exception as exc:
            logger.warning("STT failed: %s", exc)
            return ""

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
    from pc_agent.app import Agent, _setup_logging
    from pc_agent.config import get_api_key

    _setup_logging()
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
