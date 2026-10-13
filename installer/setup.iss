; Inno Setup script for PC Agent
#define MyAppName "PC Agent"
#define MyAppVersion "1.0.2"
#define MyAppPublisher "Pedro"
#define MyAppExeName "PCAgent.exe"

[Setup]
AppId={{A7C3E9D1-4B2F-4E8A-9C11-PCAGENT0001}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\PCAgent
DefaultGroupName={#MyAppName}
OutputDir=..\dist
OutputBaseFilename=PCAgent-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "autostart"; Description: "Start PC Agent when Windows starts"; GroupDescription: "Startup:"; Flags: checkedonce

[Files]
Source: "..\dist\PCAgent\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--no-browser"; Tasks: autostart

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch PC Agent"; Flags: nowait postinstall skipifsilent
