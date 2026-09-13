<#
    RLBotViewer.ps1 - the live viewer as a desktop app window.

    Starts the viewer server (hidden, pinned to CPUs 8-9, read-only on the run) if it is not already
    running, opens the viewer in its own app window (Edge app mode: no tabs or address bar, its own
    profile, and background throttling switched off), and stops the server it started when the window
    closes. Launched by the "Rocket League Bot Viewer" desktop shortcut.

        powershell -WindowStyle Hidden -ExecutionPolicy Bypass -File scripts\RLBotViewer.ps1 [-Run runs/nj3-a] [-Opponent v3]
#>
param([string]$Run = "runs/nj3-b", [ValidateSet("self", "v3", "ballchaser", "nexto")][string]$Opponent = "self", [int]$Port = 8765)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$Url = "http://127.0.0.1:$Port"
$AppData = Join-Path $env:LOCALAPPDATA "RLBotViewer"
New-Item -ItemType Directory -Force -Path $AppData | Out-Null
$Log = Join-Path $AppData "launcher.log"
function L([string]$m) { Add-Content -LiteralPath $Log -Value ((Get-Date).ToString("s") + " " + $m) }

function PortOpen { try { $c = New-Object Net.Sockets.TcpClient; $c.Connect("127.0.0.1", $Port); $c.Close(); $true } catch { $false } }

# 1. the server
$startedServer = $false
if (-not (PortOpen)) {
    L "starting viewer server for $Run (opponent $Opponent)"
    Start-Process -FilePath $Py -WorkingDirectory $Repo -WindowStyle Hidden `
        -ArgumentList "scripts\run_low.py --cpus 8,9 viewer\live_viewer.py --run $Run --opponent $Opponent --port $Port" `
        -RedirectStandardOutput (Join-Path $AppData "server.out") -RedirectStandardError (Join-Path $AppData "server.err") | Out-Null
    $startedServer = $true
    $deadline = (Get-Date).AddSeconds(90)
    while (-not (PortOpen) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
    if (-not (PortOpen)) {
        Add-Type -AssemblyName PresentationFramework
        [System.Windows.MessageBox]::Show("The viewer server did not start within 90 s.`nSee $AppData\server.err", "Rocket League Bot Viewer") | Out-Null
        exit 1
    }
} else { L "viewer server already running on $Port; using it" }

# 2. the app window
$browsers = @("${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
              "$env:ProgramFiles\Google\Chrome\Application\chrome.exe", "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe")
$exe = $browsers | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) { Start-Process $Url; exit 0 }
$profileDir = Join-Path $AppData "window-profile"
$flags = @("--app=$Url", "--user-data-dir=`"$profileDir`"", "--window-size=1600,960", "--no-first-run", "--no-default-browser-check",
           "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows",
           "--ignore-gpu-blocklist", "--enable-gpu-rasterization")
L "opening app window with $(Split-Path -Leaf $exe)"
Start-Process -FilePath $exe -ArgumentList $flags | Out-Null

# 3. wait for the window to close (every process of the app's own profile gone), then stop the server we started
Start-Sleep -Seconds 5
$escaped = [regex]::Escape($profileDir)
while (@(Get-CimInstance Win32_Process -Filter "Name='msedge.exe' OR Name='chrome.exe'" | Where-Object { $_.CommandLine -match $escaped }).Count -gt 0) {
    Start-Sleep -Seconds 2
}
L "app window closed"
if ($startedServer) {
    foreach ($p in @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'live_viewer\.py' -and $_.CommandLine -match "--port $Port" })) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    L "viewer server stopped"
}
