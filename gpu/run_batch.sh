#!/bin/bash
# desktop 배치가 ssh 로 부른다. batches/<id>/request.jsonl 의 논문 HTML 을 받아 전문·그림을 뽑고 GPU job 을 넣는다.
# HTML 받기는 네트워크만 쓰므로 GPU job 밖에서 먼저 끝낸다(논문당 3초 간격이라 600편이면 30분).
# 진행 상태는 batches/<id>/status 한 줄: extracting → queued <job> → running <job> → done | failed
# 사용: bash run_batch.sh <batch_id>
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$ROOT/server.env"
B=$ROOT/batches/$1
[ -f "$B/request.jsonl" ] || { echo "no request: $B" >&2; exit 1; }

# ssh 재시도로 두 번 불려도 한 번만 돈다
if ! (set -o noclobber; echo extracting > "$B/status") 2>/dev/null; then
  echo "already started: $(cat "$B/status")"; exit 0
fi
if ! "$PY" "$ROOT/extract.py" "$B/request.jsonl" "$B/papers.jsonl"; then
  echo failed > "$B/status"; exit 1
fi

cd "$ROOT"
if ! JOB=$(BATCH_DIR=$B sbatch --parsable --nice="${NICE:-0}" --time="${JOB_TIME:-4:00:00}" \
    --partition="${PARTITION:-gpu}" --output="$B/job-%j.out" --export=ALL job.sh); then
  echo failed > "$B/status"; exit 1
fi
echo "queued $JOB" > "$B/status"
echo "submitted $JOB"
