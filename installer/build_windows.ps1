$ErrorActionPreference = "Stop"
python -m pip install -U pip
pip install -e ".[windows,voice]"
pip install pyinstaller pyttsx3 pywin32 sounddevice

pyinstaller --noconfirm --clean `
  --name PCAgent `
  --windowed `
  --collect-all google.genai `
  --collect-all keyring `
  --hidden-import=pystray._win32 `
  --hidden-import=win32timezone `
  --hidden-import=uvicorn.logging `
  --hidden-import=uvicorn.loops `
  --hidden-import=uvicorn.loops.auto `
  --hidden-import=uvicorn.protocols `
  --hidden-import=uvicorn.protocols.http `
  --hidden-import=uvicorn.protocols.http.auto `
  --hidden-import=uvicorn.protocols.websockets `
  --hidden-import=uvicorn.protocols.websockets.auto `
  --hidden-import=uvicorn.lifespan `
  --hidden-import=uvicorn.lifespan.on `
  --hidden-import=pc_agent.core.voice `
  --hidden-import=pc_agent.core.live_session `
  --hidden-import=pc_agent.ui.voice_hub `
  --hidden-import=pc_agent.core.hotkeys `
  --collect-submodules pc_agent `
  src/pc_agent/__main__.py

if (-not (Test-Path "dist/PCAgent/PCAgent.exe")) { throw "PyInstaller output missing" }

& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\setup.iss
if (-not (Test-Path "dist/PCAgent-Setup.exe")) { throw "Installer missing" }
Write-Host "Build OK"
