# arxivbot on-demand 리스너를 부팅 시 자동 실행하도록 Task Scheduler에 등록.
# 실행: 관리자 PowerShell에서  powershell -ExecutionPolicy Bypass -File scripts\install_listener_task.ps1
#
# 작업이 python을 직접 띄우면 Microsoft Store판 python 별칭이 비대화형 컨텍스트에서
# 실행되지 않아 즉시 종료된다. 그래서 ① 등록 시점(대화형)에 python.exe 절대경로를 확보하고
# ② powershell 래퍼(run_listener.ps1)로 실행해 모든 출력을 logs\listener.log에 남긴다.
#
# 리스너는 Socket Mode라 데스크톱이 필요 없으므로 S4U(로그온 여부 무관)로 돌린다.
# 로그온 트리거만 있으면 Windows Update 재부팅처럼 아무도 로그인하지 않는 재시작 뒤에
# 영영 안 뜬다 -> 부팅 트리거가 본체이고 로그온 트리거는 보조다.
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

# 부팅 시 시작 + 로그온 시 시작(보조) + 30분마다 워치독.
# 워치독은 MultipleInstances=IgnoreNew 덕에 이미 돌고 있으면 아무 일도 하지 않고,
# 프로세스가 조용히 죽어 있었을 때만 다시 띄운다(재시작 정책이 못 잡는 정상 종료 대비).
$triggers = @(
    New-ScheduledTaskTrigger -AtStartup
    New-ScheduledTaskTrigger -AtLogOn
    New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 30)
)

# S4U: 비밀번호 저장 없이 "로그온 여부에 관계없이 실행". 일별 배치 작업과 동일한 방식.
$principal = New-ScheduledTaskPrincipal `
    -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType S4U -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 9999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $triggers `
    -Principal $principal -Settings $settings -Force `
    -Description "arxivbot on-demand summary Socket Mode listener"

Write-Host "Registered '$taskName'."
Write-Host "Logs  : $repo\logs\listener.log"
Write-Host "Start : Start-ScheduledTask -TaskName $taskName"
