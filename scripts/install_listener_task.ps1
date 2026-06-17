# arxivbot on-demand 리스너를 로그온 시 자동 실행하도록 Task Scheduler에 등록.
# 실행: 관리자 PowerShell에서  powershell -ExecutionPolicy Bypass -File scripts\install_listener_task.ps1
$ErrorActionPreference = "Stop"

$repo     = "C:\Users\hist0\Dropbox\develop\arxivbot_new"
$python   = "python"   # 배치 main.py와 동일한 Python 3.11. 필요하면 절대경로로.
$taskName = "arxivbot-listener"

$action = New-ScheduledTaskAction -Execute $python -Argument "listener.py" -WorkingDirectory $repo
# 로그온 시 시작(사용자 컨텍스트라 .env/네트워크/PATH가 안정적).
# 사용자 로그인 없이 부팅만으로 띄우려면 -AtStartup 으로 바꾸고 -User/-Password 또는 SYSTEM 사용.
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

Write-Host "Registered '$taskName' (AtLogOn, restart every 1 min on failure)."
Write-Host "Start now:  Start-ScheduledTask -TaskName $taskName"
