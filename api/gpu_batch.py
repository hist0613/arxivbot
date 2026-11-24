"""배치 요약을 GPU 서버 slurm job 에 맡기고, 끝난 결과를 받아 캐시에 넣는다.

흐름(배치 한 번 실행마다):
1. collect(): 보내 둔 배치 가운데 서버에서 끝난 것의 results.jsonl 과 그림을 받아 summaries·figures 에 넣는다.
2. 크롤.
3. submit(): 요약이 없고 아직 보내지 않은 논문을 한 묶음으로 서버에 올리고 run_batch.sh 를 띄운다.
4. 요약이 있는 논문만 게시한다. 나머지는 결과가 오는 다음 실행 때 게시된다.

job 은 카드가 빌 때까지 기다리므로 몇 시간 늦을 수 있다. 서버 주소·경로는 코드에 두지 않고
settings 의 GPU_SSH_HOST(ssh config Host 별칭), GPU_REMOTE_DIR 로 받는다.
"""

import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from api.arxiv import parse_arxiv_ref
from api.logger import logger

REPO_DIR = Path(__file__).resolve().parent.parent
# 서버에 올리는 파일. server.env 는 서버에만 있다.
DEPLOY_FILES = ["gpu/extract.py", "gpu/summarize.py", "gpu/preflight.py",
                "gpu/job.sh", "gpu/run_batch.sh", "prompts.py"]

DISPLAY_KEYS = {
    "prior_approaches": "Prior Approaches",
    "core_contribution": "Core Contribution",
    "technical_challenges": "Technical Challenges",
    "empirical_impact": "Empirical Impact",
}


def text_to_sections(text: str) -> dict:
    """서버 추출 전문(제목 줄이 "## ...")을 배치 크롤과 같은 {id: {title, content}} 모양으로 나눈다.

    배치 크롤(api/arxiv.py get_paper_full_content)이 쓰는 모양이라 OpenAI 배치로 되돌려도 그대로 읽힌다.
    제목 앞부분(초록 등)은 "Front matter" 로 둔다. 그림·표 캡션은 본문 안에 [...] 줄로 남아 있다.
    """
    sections, title, lines = {}, "Front matter", []

    def flush():
        content = "\n".join(l for l in lines if l.strip()).strip()
        if content:
            sections[f"S{len(sections)}"] = {"title": title, "content": content}

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            title, lines = line[3:].strip() or "No title found", []
        else:
            lines.append(line)
    flush()
    return sections


def to_cached_summary(summary: dict) -> str:
    """서버 결과(영어 소문자 키)를 배치 캐시와 같은 표시 키 JSON 으로. api/agent.py _summary_to_dict 와 같은 모양."""
    return json.dumps({DISPLAY_KEYS[k]: summary.get(k, "") for k in DISPLAY_KEYS})


class Remote:
    """ssh/scp 호출. 테스트에서 바꿔 끼운다.

    공개 포트의 sshd 는 무차별 대입 시도로 연결 한도가 차면 새 연결을 확률적으로 끊는다
    (kex_exchange_identification). 그래서 실패하면 잠시 뒤 다시 시도한다.
    """

    RETRIES = 4
    RETRY_SEC = 20

    def __init__(self, host: str, remote_dir: str):
        self.host = host
        self.remote_dir = remote_dir.rstrip("/")

    def _call(self, args: list, timeout: int) -> str:
        err = ""
        for attempt in range(self.RETRIES):
            try:
                result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                err = f"timeout {timeout}s"
            else:
                if result.returncode == 0:
                    return result.stdout
                err = f"{result.returncode}: {result.stderr.strip()[:300]}"
            if attempt < self.RETRIES - 1:
                time.sleep(self.RETRY_SEC)
        raise RuntimeError(f"{args[0]} 실패({err})")

    def run(self, command: str, timeout: int = 120) -> str:
        return self._call(["ssh", "-o", "BatchMode=yes", self.host, f"cd {self.remote_dir} && {command}"], timeout)

    def put(self, local: str, remote_rel: str, timeout: int = 300):
        self._scp([local, f"{self.host}:{self.remote_dir}/{remote_rel}"], timeout)

    def get(self, remote_rel: str, local: str, recursive: bool = False, timeout: int = 600):
        self._scp((["-r"] if recursive else []) + [f"{self.host}:{self.remote_dir}/{remote_rel}", local], timeout)

    def _scp(self, args: list, timeout: int):
        self._call(["scp", "-q", "-C", "-o", "BatchMode=yes"] + args, timeout)


MAX_ATTEMPTS = 2  # 두 번 보내도 요약이 안 온 논문은 더 보내지 않는다(게시도 되지 않는다)


class GpuBatch:
    def __init__(self, cache, remote: Remote, state_dir: str, figure_dir: str, max_age_hours: float = 36):
        self.cache = cache
        self.remote = remote
        self.state_dir = Path(state_dir)
        self.figure_dir = Path(figure_dir)
        self.max_age_sec = max_age_hours * 3600
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "batches.json"

    # --- 상태 ---

    def _load(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {}

    def _save(self, state: dict):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def pending_infos(self) -> set:
        return {info for b in self._load().values() if b["status"] == "submitted" for info in b["papers"]}

    def failed_attempts(self) -> dict:
        """끝난(받았거나 버린) 배치에 들어 있었는데 요약이 오지 않은 횟수. 계속 실패하는 논문을 매번 다시 보내지 않으려고."""
        counts = {}
        for b in self._load().values():
            if b["status"] in ("collected", "abandoned"):
                for info in set(b["papers"]) - set(b.get("imported", [])):
                    counts[info] = counts.get(info, 0) + 1
        return counts

    # --- 1. 결과 받기 ---

    def collect(self) -> int:
        """끝난 배치를 받아 캐시에 넣는다. 넣은 요약 수를 돌려준다."""
        state = self._load()
        imported = 0
        for batch_id, batch in state.items():
            if batch["status"] != "submitted":
                continue
            try:
                status = self.remote.run(f"cat batches/{batch_id}/status 2>/dev/null || echo missing").strip()
            except Exception as e:
                logger.warning(f"[gpu] {batch_id} 상태 확인 실패: {e}")
                continue
            if status == "done":
                local = self.state_dir / batch_id
                local.mkdir(exist_ok=True)
                try:
                    self.remote.get(f"batches/{batch_id}/results.jsonl", str(local / "results.jsonl"))
                    if self.remote.run(f"test -d batches/{batch_id}/figs && echo yes || echo no").strip() == "yes":
                        self.remote.get(f"batches/{batch_id}/figs", str(local), recursive=True)
                except Exception as e:
                    logger.warning(f"[gpu] {batch_id} 결과 받기 실패: {e}")
                    continue
                done_infos = self.import_results(local / "results.jsonl", local / "figs")
                self._collect_full_texts(batch_id, local)
                n = len(done_infos)
                imported += n
                batch["imported"] = done_infos
                batch["status"] = "collected"
                batch["collected_at"] = time.time()
                logger.info(f"[gpu] {batch_id} 받음: 요약 {n}/{len(batch['papers'])}편")
            elif status == "failed" or time.time() - batch["submitted_at"] > self.max_age_sec:
                batch["status"] = "abandoned"
                logger.warning(f"[gpu] {batch_id} 버림({status}). 그 논문들은 다음 submit 때 다시 보낸다")
            else:
                logger.info(f"[gpu] {batch_id} 진행 중: {status}")
        self._save(state)
        return imported

    def _collect_full_texts(self, batch_id: str, local: Path):
        """서버가 뽑은 전문을 받아 full_contents 에 넣는다. 그림 파일은 받지 않는다(대표 그림만 figs/ 로 온다).
        못 받아도 요약 게시에는 지장이 없으므로 경고만 남긴다."""
        path = local / "papers.jsonl"
        try:
            self.remote.get(f"batches/{batch_id}/papers.jsonl", str(path))
            n = self.import_full_texts(path)
            logger.info(f"[gpu] {batch_id} 본문 {n}편 저장")
        except Exception as e:
            logger.warning(f"[gpu] {batch_id} 본문 받기 실패: {e}")
        finally:
            # 본문은 DB 에 들어갔으니 사본은 남기지 않는다(하루 수십 MB)
            path.unlink(missing_ok=True)

    def import_full_texts(self, papers_path: Path) -> int:
        n = 0
        for line in papers_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            info, text = rec.get("paper_info"), rec.get("text")
            if not info or not text or info in self.cache.paper_full_contents:
                continue
            self.cache.update_paper_full_contents(info, text_to_sections(text))
            n += 1
        return n

    def import_results(self, results_path: Path, figs_dir: Path) -> list:
        """요약을 넣은 paper_info 목록을 돌려준다."""
        done = []
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            info, summary = rec.get("paper_info"), rec.get("summary")
            if not info or not summary or not all(summary.get(k) for k in DISPLAY_KEYS):
                continue
            self.cache.update_paper_summarizations(info, to_cached_summary(summary), model=rec.get("model", ""))
            done.append(info)
            fig = rec.get("figure") or {}
            src = figs_dir / f"{rec['arxiv_id']}.png"
            if fig.get("figure_id") and src.exists():
                dst = self.figure_dir / f"{rec['arxiv_id']}.png"
                shutil.copyfile(src, dst)
                label = fig.get("label") or fig["figure_id"]
                self.cache.store.put_figure(info, str(dst), label, fig.get("description", ""), rec.get("model", ""))
        return done

    # --- 3. 보내기 ---

    def deploy(self):
        for rel in DEPLOY_FILES:
            self.remote.put(str(REPO_DIR / rel), Path(rel).name)

    def submit(self, papers: list) -> str | None:
        """papers: (paper_url, paper_title, paper_info) 목록. 요약이 이미 있거나 보내 둔 논문은 뺀다."""
        pending = self.pending_infos()
        attempts = self.failed_attempts()
        rows, seen = [], set()
        for url, title, info in papers:
            if info in seen or info in pending or self.cache.has_paper_summarization(info):
                continue
            if attempts.get(info, 0) >= MAX_ATTEMPTS:
                continue
            ref = parse_arxiv_ref(url)
            if not ref:
                continue
            seen.add(info)
            rows.append({"arxiv_id": ref.rsplit("/", 1)[-1], "paper_info": info, "title": title, "url": url,
                         "abstract": self.cache.paper_abstracts.get(info, "")})
        if not rows:
            return None
        batch_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        state = self._load()
        while batch_id in state:
            batch_id += "x"
        local = self.state_dir / batch_id
        local.mkdir(parents=True, exist_ok=True)
        request = local / "request.jsonl"
        request.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        self.deploy()
        self.remote.run(f"mkdir -p batches/{batch_id}")
        self.remote.put(str(request), f"batches/{batch_id}/request.jsonl")
        # HTML 받기가 수십 분 걸리므로 서버에서 떼어 놓고 돌린다
        self.remote.run(f"nohup bash run_batch.sh {batch_id} > batches/{batch_id}/run.log 2>&1 < /dev/null &")
        state[batch_id] = {"status": "submitted", "submitted_at": time.time(), "papers": [r["paper_info"] for r in rows]}
        self._save(state)
        logger.info(f"[gpu] {batch_id} 보냄: {len(rows)}편")
        return batch_id
