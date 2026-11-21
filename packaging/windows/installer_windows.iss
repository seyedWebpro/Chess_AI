; Inno Setup Script for Chess AI Multi-Modal Grandmaster System
; Generates standard single-click Windows setup wizard (ChessAI_Setup.exe)

[Setup]
AppId={{C4E55A12-88B1-4A53-B041-A2981CF4F56E}
AppName=Chess AI Grandmaster Assistant
AppVersion=2.0.0
AppPublisher=Chess AI Open Collective
DefaultDirName={autopf}\ChessAI
DefaultGroupName=Chess AI
OutputDir=..\..\dist
OutputBaseFilename=ChessAI_Setup_v2.0.0_x64
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\..\dist\ChessAI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Chess AI"; Filename: "{app}\ChessAI.exe"
Name: "{group}\{cm:UninstallProgram,Chess AI}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Chess AI"; Filename: "{app}\ChessAI.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ChessAI.exe"; Description: "{cm:LaunchProgram,Chess AI}"; Flags: nowait postinstall skipifsilent
