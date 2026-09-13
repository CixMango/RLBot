<#
    install_viewer_shortcut.ps1 - put a "Rocket League Bot Viewer" shortcut on the Desktop and in the Start menu.

    The shortcut runs scripts\RLBotViewer.ps1 with no console window. Remove it by deleting the two .lnk files.

        powershell -ExecutionPolicy Bypass -File scripts\install_viewer_shortcut.ps1
#>
$Repo = Split-Path -Parent $PSScriptRoot
$target = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$lnkArgs = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Repo\scripts\RLBotViewer.ps1`""
$edge = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
$icon = if (Test-Path $edge) { "$edge,0" } else { "$env:SystemRoot\System32\shell32.dll,17" }
$shell = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath("Desktop"), (Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs"))) {
    $lnk = $shell.CreateShortcut((Join-Path $dir "Rocket League Bot Viewer.lnk"))
    $lnk.TargetPath = $target
    $lnk.Arguments = $lnkArgs
    $lnk.WorkingDirectory = $Repo
    $lnk.IconLocation = $icon
    $lnk.WindowStyle = 7
    $lnk.Description = "Live viewer for the no-jump bot's training run"
    $lnk.Save()
    Write-Host "created $(Join-Path $dir 'Rocket League Bot Viewer.lnk')"
}
