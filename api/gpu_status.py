"""`/gpu` slash command — 랩 GPU 서버(hufs5090, hufs4090)의 지금 상태를 모아 온다.

서버마다 dilab `tools/gpu-slack-bot`이 10분마다 #server에 리포트를 올리는데, 그
사이의 상태가 궁금할 때 쓰는 명령이다. 리포트 형식을 여기서 다시 만들지 않고 서버의
`monitor.py --report` 출력을 ssh로 받아 그대로 붙인다.

ssh 대상은 `~/.ssh/config`의 Host 이름이다. 이 키는 서버 `authorized_keys`에서
`monitor.py --report`만 실행되게 묶여 있어서, 어떤 명령을 보내도 리포트만 돌아온다.
"""
import subprocess
from concurrent.futures import ThreadPoolExecutor


def fetch_report(host, runner=subprocess.run, attempts=2):
    """서버 하나의 리포트. 끝내 못 받으면 그 서버 자리에 경고를 남긴다.

    hufs5090의 ssh 포트는 무차별 대입이 늘 들어와서 새 연결이 가끔
    `kex_exchange_identification: Connection reset`으로 끊긴다. 다시 붙으면 되므로
    한 번 더 시도한다.
    """
    detail = ""
    for _ in range(attempts):
        try:
            result = runner(
                ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                 host, "report"],
                capture_output=True,
                text=True,
                # 서버 출력은 UTF-8(이모지 포함)인데 Windows 기본 코덱은 cp949다.
                encoding="utf-8",
                errors="replace",
                timeout=40,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            detail = str(e)
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        detail = (result.stderr or result.stdout or "").strip() or f"rc={result.returncode}"
    return f"⚠️ [{host}] 상태를 받아오지 못했습니다\n```\n{detail[:300]}\n```"


def gather_reports(hosts, fetch=fetch_report):
    """서버 리포트를 동시에 받아 설정 순서대로 붙인다."""
    if not hosts:
        return "조회할 GPU 서버가 설정돼 있지 않습니다(settings.py gpu_status_hosts)."
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        reports = list(pool.map(fetch, hosts))
    return "\n".join(reports)
