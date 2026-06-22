# arxivbot on-demand 리스너를 로그온 시 자동 실행하도록 Task Scheduler에 등록.
# 실행: 관리자 PowerShell에서  powershell -ExecutionPolicy Bypass -File scripts\install_listener_task.ps1
#
# 작업이 python을 직접 띄우면 Microsoft Store판 python 별칭이 비대화형 컨텍스트에서
# 실행되지 않아 즉시 종료된다. 그래서 ① 등록 시점(대화형)에 python.exe 절대경로를 확보하고
# ② powershell 래퍼(run_listener.ps1)로 실행해 모든 출력을 logs\listener.log에 남긴다.
$ErrorActionPreference = "Stop"

$repo     = Split-Path -Parent $PSScriptRoot
$taskName = "arxivbot-listener"

$python = (& python -c "import sys; print(sys.executable)") 2>$null
if (-not $python) {
    throw "python을 찾지 못했습니다. 이 PowerShell에서 'python --version'이 되는지 확인하세요."
}
$python = $python.Trim()
Write-Host "python.exe = $python"

$wrapper  = Join-Path $repo "scripts\run_listener.ps1"
$argument = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$wrapper`" -Python `"$python`""

$action  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument -WorkingDirectory $repo
# 로그온 시 시작(사용자 컨텍스트라 .env/네트워크/PATH가 안정적).
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 9999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -RunLevel Highest -Force `
    -Description "arxivbot on-demand summary Socket Mode listener"

Write-Host "Registered '$taskName'."
Write-Host "Logs  : $repo\logs\listener.log"
Write-Host "Start : Start-ScheduledTask -TaskName $taskName"
