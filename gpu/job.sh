#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --job-name=arxivbot
# arxivbot 요약 job: vLLM 을 띄우고 summarize.py 로 논문을 처리한 뒤 vLLM 을 프로세스 그룹째 죽인다.
#
# 서버마다 다른 값(파이썬 경로, 모델, 피할 GPU 번호)은 이 폴더의 server.env 에 둔다(git 에 올리지 않는다).
# 예시는 server.env.example.
#
# 두 가지로 쓴다.
#   매일 배치: run_batch.sh 가 BATCH_DIR 을 주고 제출한다. 결과는 $BATCH_DIR/results.jsonl, 그림은 $BATCH_DIR/figs/.
#   비교 실험: cd <이 폴더> && PAPERS=... RUN_TAG=... sbatch --output=logs/%x-%j.out job.sh
#             결과는 results/$RUN_TAG/$SERVED.jsonl. MODEL_ID 등을 환경변수로 주면 server.env 값보다 우선한다.
set -uo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
cd "$ROOT"
OVERRIDE_MODEL_ID=${MODEL_ID:-}; OVERRIDE_SERVED=${SERVED:-}; OVERRIDE_PARSER=${REASONING_PARSER:-}; OVERRIDE_VLLM_PY=${VLLM_PY:-}
# shellcheck disable=SC1091
source "$ROOT/server.env"
MODEL_ID=${OVERRIDE_MODEL_ID:-$MODEL_ID}; SERVED=${OVERRIDE_SERVED:-$SERVED}
REASONING_PARSER=${OVERRIDE_PARSER:-$REASONING_PARSER}; VLLM_PY=${OVERRIDE_VLLM_PY:-$VLLM_PY}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
CONCURRENCY=${CONCURRENCY:-8}
PORT=$((19000 + SLURM_JOB_ID % 1000))

if [ -n "${BATCH_DIR:-}" ]; then
  PAPERS=$BATCH_DIR/papers.jsonl
  OUT=$BATCH_DIR/results.jsonl
  FIGS_OUT=(--figs-out "$BATCH_DIR/figs")
  set_status() { echo "$1" > "$BATCH_DIR/status"; }
else
  PAPERS=${PAPERS:-$ROOT/papers.jsonl}
  OUT=$ROOT/results/${RUN_TAG:-bench}/$SERVED.jsonl
  FIGS_OUT=()
  set_status() { :; }
fi
mkdir -p "$(dirname "$OUT")" "$ROOT/logs"

echo "[job] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES model=$MODEL_ID port=$PORT start=$(date -Is)"

# AVOID_GPUS 에 적은 카드(예: 다른 사람 전용 카드)를 받으면 내려가고 20분 뒤로 다시 낸다.
# slurm gres 에 카드 번호를 고르는 방법이 없어서 이렇게 한다.
if [ -n "${AVOID_GPUS:-}" ]; then
  for g in ${AVOID_GPUS//,/ }; do
    if echo ",$CUDA_VISIBLE_DEVICES," | grep -q ",$g,"; then
      NEW=$(sbatch --parsable --begin=now+20minutes --nice="${NICE:-0}" --time="${JOB_TIME:-4:00:00}" \
        --partition="${PARTITION:-gpu}" --output="$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*StdOut=//p')" \
        --export=ALL "$ROOT/job.sh")
      echo "[job] GPU $CUDA_VISIBLE_DEVICES 에 $g 가 있어 쓰지 않음, job $NEW 로 다시 제출"
      set_status "queued $NEW"
      exit 0
    fi
  done
fi

set_status "running $SLURM_JOB_ID"
export MODEL_ID REASONING_PARSER PAPERS HF_EXTRA_CACHES
"$PY" "$ROOT/preflight.py" || { set_status failed; exit 1; }

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# vLLM 은 기동 중 커널을 JIT 컴파일하며 ninja·nvcc 를 PATH 에서 찾는다
export PATH="$(dirname "$VLLM_PY"):${EXTRA_PATH:+$EXTRA_PATH:}$CUDA_HOME/bin:$PATH"

PGID=""
kill_group() {
  [ -n "$PGID" ] || return
  kill -TERM -"$PGID" 2>/dev/null; sleep 5; kill -KILL -"$PGID" 2>/dev/null
  PGID=""
}
trap kill_group EXIT

export VLLM_API_KEY=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d " \n")
setsid "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" --served-model-name "$SERVED" --port "$PORT" \
  --tensor-parallel-size "${TP:-2}" --gpu-memory-utilization 0.90 --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$CONCURRENCY" --reasoning-parser "$REASONING_PARSER" \
  --limit-mm-per-prompt '{"image": 3, "video": 0}' --seed 0 \
  > "$ROOT/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
PGID=$!
echo "[job] vllm pgid=$PGID"

for i in $(seq 1 120); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null; then echo "[job] vllm up after $((i*10))s"; break; fi
  if ! kill -0 "$PGID" 2>/dev/null; then echo "[job] vllm died"; tail -40 "$ROOT/logs/vllm-$SLURM_JOB_ID.log"; set_status failed; exit 1; fi
  sleep 10
done
curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "[job] vllm not up"; tail -40 "$ROOT/logs/vllm-$SLURM_JOB_ID.log"; set_status failed; exit 1; }

"$PY" summarize.py --papers "$PAPERS" --out "$OUT" --model "$SERVED" \
  --base-url "http://localhost:$PORT/v1" --concurrency "$CONCURRENCY" --fig-cache "$ROOT/figcache" \
  --prompt "${PROMPT:-v2}" "${FIGS_OUT[@]}"
code=$?
if [ $code -eq 0 ]; then set_status done; else set_status failed; fi
echo "[job] done code=$code $(date -Is)"
