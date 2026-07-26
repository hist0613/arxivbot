# listener.py를 로그 파일과 함께 실행하는 래퍼.
# Task Scheduler가 이 스크립트를 호출한다(콘솔이 없으므로 모든 출력을 파일로 남긴다).
# 수동으로도 실행 가능: powershell -ExecutionPolicy Bypass -File scripts\run_listener.ps1
param([string]$Python = "python")

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$log = Join-Path $logDir "listener.log"

"==== $(Get-Date -Format o) starting (python=$Python, cwd=$repo) ====" |
    Out-File -FilePath $log -Append -Encoding utf8

# Stop-ScheduledTask는 이 래퍼만 끝내고 python 자식은 고아로 남긴다. 그대로 두면
# Socket Mode 연결이 둘이 되어 멘션이 옛 프로세스로 배정되기도 하므로, 새로 뜨기
# 전에 지난 인스턴스를 정리한다. MultipleInstances=IgnoreNew라 정상 동작 중인
# 인스턴스가 있으면 이 래퍼 자체가 실행되지 않는다.
$pidFile = Join-Path $logDir "listener.pid"
if (Test-Path $pidFile) {
    $stalePid = (Get-Content $pidFile -Raw).Trim()
    $stale = $null
    if ($stalePid -match '^\d+$') {
        $stale = Get-Process -Id ([int]$stalePid) -ErrorAction SilentlyContinue
    }
    # PID는 재사용되므로 이름까지 확인하고 죽인다.
    if ($stale -and $stale.ProcessName -like "python*") {
        "  killing stale listener pid=$stalePid" |
            Out-File -FilePath $log -Append -Encoding utf8
        Stop-Process -Id $stalePid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
}

# -u: 출력 버퍼링 끄기(장시간 프로세스의 로그가 즉시 파일에 쌓이도록).
# *>> : stdout/stderr/모든 스트림(트레이스백 포함)을 로그에 append.
& $Python -u listener.py *>> $log
$code = $LASTEXITCODE

"==== $(Get-Date -Format o) EXITED code=$code ====" |
    Out-File -FilePath $log -Append -Encoding utf8

# 종료 코드를 그대로 넘겨야 Task Scheduler의 실패 시 재시작 정책이 동작한다.
exit $code
