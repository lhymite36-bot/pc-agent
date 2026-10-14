; Inno Setup script for PC Agent / Atlas
#define MyAppName "PC Agent"
#define MyAppVersion "1.0.11"
#define MyAppPublisher "Pedro"
#define MyAppExeName "PCAgent.exe"
#define MyCliExeName "AtlasCLI.exe"
#define MyAppMutex "PCAgentSingletonMutex_A7C3E9D1"

[Setup]
AppId={{A7C3E9D1-4B2F-4E8A-9C11-PCAGENT0001}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppMutex={#MyAppMutex}
DefaultDirName={localappdata}\Programs\PCAgent
DefaultGroupName={#MyAppName}
OutputDir=..\dist
OutputBaseFilename=PCAgent-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
CloseApplicationsFilter=PCAgent.exe,AtlasCLI.exe
RestartApplications=yes
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "autostart"; Description: "Start PC Agent when Windows starts"; GroupDescription: "Startup:"; Flags: checkedonce

[Files]
Source: "..\dist\PCAgent\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Atlas (Terminal)"; Filename: "{app}\{#MyCliExeName}"; Comment: "Atlas in your terminal with mic voice"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{autodesktop}\Atlas (Terminal)"; Filename: "{app}\{#MyCliExeName}"; Tasks: desktopicon
Name: "{userstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--no-browser"; Tasks: autostart

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch PC Agent"; Flags: nowait postinstall skipifsilent
Filename: "{app}\{#MyCliExeName}"; Description: "Launch Atlas (Terminal)"; Flags: nowait postinstall skipifsilent unchecked

[Code]
function KillImage(const Name: string): Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('taskkill.exe', '/F /IM ' + Name, '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  KillImage('PCAgent.exe');
  KillImage('AtlasCLI.exe');
  Sleep(800);
end;

function InitializeSetup(): Boolean;
begin
  // Best-effort: close running instances before file copy
  KillImage('PCAgent.exe');
  KillImage('AtlasCLI.exe');
  Result := True;
end;
