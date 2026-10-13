$ErrorActionPreference = "Stop"
python -m pip install -U pip
pip install -e ".[windows,voice]"
pip install pyinstaller pyttsx3 pywin32 sounddevice rich numpy faster-whisper

$common = @(
  "--noconfirm", "--clean",
  "--collect-all", "google.genai",
  "--collect-all", "keyring",
  "--collect-all", "sounddevice",
  "--collect-all", "rich",
  "--hidden-import=pystray._win32",
  "--hidden-import=win32timezone",
  "--hidden-import=uvicorn.logging",
  "--hidden-import=uvicorn.loops",
  "--hidden-import=uvicorn.loops.auto",
  "--hidden-import=uvicorn.protocols",
  "--hidden-import=uvicorn.protocols.http",
  "--hidden-import=uvicorn.protocols.http.auto",
  "--hidden-import=uvicorn.protocols.websockets",
  "--hidden-import=uvicorn.protocols.websockets.auto",
  "--hidden-import=uvicorn.lifespan",
  "--hidden-import=uvicorn.lifespan.on",
  "--hidden-import=pc_agent.core.voice",
  "--hidden-import=pc_agent.core.live_session",
  "--hidden-import=pc_agent.core.terminal_audio",
  "--hidden-import=pc_agent.cli",
  "--hidden-import=pc_agent.ui.voice_hub",
  "--hidden-import=pc_agent.core.hotkeys",
  "--hidden-import=sounddevice",
  "--hidden-import=faster_whisper",
  "--collect-all", "faster_whisper",
  "--hidden-import=_sounddevice_data",
  "--collect-submodules", "pc_agent",
  "--collect-binaries", "sounddevice"
)

# Windowed tray/web app
pyinstaller @common --name PCAgent --windowed src/pc_agent/__main__.py
if (-not (Test-Path "dist/PCAgent/PCAgent.exe")) { throw "PCAgent.exe missing" }

# Console terminal app (shares _internal with onedir layout — build into same dist folder)
pyinstaller @common --name AtlasCLI --console --distpath dist/PCAgent_cli_build src/pc_agent/cli_main.py
if (-not (Test-Path "dist/PCAgent_cli_build/AtlasCLI/AtlasCLI.exe")) { throw "AtlasCLI.exe missing" }

# Merge AtlasCLI into PCAgent onedir so one install has both
Copy-Item -Force "dist/PCAgent_cli_build/AtlasCLI/AtlasCLI.exe" "dist/PCAgent/AtlasCLI.exe"
# Copy any extra DLLs from CLI build that PCAgent tree might lack (PortAudio etc.)
Get-ChildItem "dist/PCAgent_cli_build/AtlasCLI" -File | Where-Object { $_.Extension -in ".dll",".pyd" } | ForEach-Object {
  $dest = Join-Path "dist/PCAgent" $_.Name
  if (-not (Test-Path $dest)) { Copy-Item -Force $_.FullName $dest }
}
# Nested dirs (e.g. sounddevice PortAudio data)
Get-ChildItem "dist/PCAgent_cli_build/AtlasCLI" -Directory | ForEach-Object {
  $dest = Join-Path "dist/PCAgent" $_.Name
  if (-not (Test-Path $dest)) {
    Copy-Item -Recurse -Force $_.FullName $dest
  } else {
    Copy-Item -Recurse -Force (Join-Path $_.FullName "*") $dest -ErrorAction SilentlyContinue
  }
}

# Smoke test frozen CLI
& "dist/PCAgent/AtlasCLI.exe" --selftest
if ($LASTEXITCODE -ne 0) { throw "AtlasCLI --selftest failed with $LASTEXITCODE" }

& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\setup.iss
if (-not (Test-Path "dist/PCAgent-Setup.exe")) { throw "Installer missing" }
Write-Host "Build OK"
