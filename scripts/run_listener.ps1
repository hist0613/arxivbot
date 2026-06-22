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

# -u: 출력 버퍼링 끄기(장시간 프로세스의 로그가 즉시 파일에 쌓이도록).
# *>> : stdout/stderr/모든 스트림(트레이스백 포함)을 로그에 append.
& $Python -u listener.py *>> $log

"==== $(Get-Date -Format o) EXITED code=$LASTEXITCODE ====" |
    Out-File -FilePath $log -Append -Encoding utf8
