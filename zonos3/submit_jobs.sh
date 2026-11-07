#!/bin/bash
# HF Jobs TTS eval — ZONOS3 Mini backend (generate → transcribe → SIM → UTMOS → score), speaker-embedding cloning.
# Runs locally; submits HF Jobs. Usage: bash zonos3/submit_jobs.sh ["Zyphra/ZONOS3-Mini <max_running> hf://..." ...]
# The runtime (github.com/Zyphra/ZONOS3) and the weights (Zyphra/ZONOS3_Mini_release_ckpts) are private: stage 1 runs
# on a public CUDA base image, clones the pinned runtime with ZONOS3_GITHUB_TOKEN (from ZONOS3_SECRETS_FILE) and
# pulls the weights with HF_TOKEN. Shared orchestration (stages 2-4, injection, logging, scoring) lives in
# scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
ZONOS3_IMAGE="${ZONOS3_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}"   # CUDA 12.8 toolkit: the runtime JIT-compiles kernels
ZONOS3_REPO="https://github.com/Zyphra/ZONOS3.git"
ZONOS3_REV="${ZONOS3_REV:-1c551947ebc996395d737276f3b4a0ba031eac09}"   # leaderboard-throughput (main a884139 + server throughput fixes): release LM rate30_20k_r2grpo, KEEP-2 decoder
ZONOS3_SECRETS_FILE="${ZONOS3_SECRETS_FILE:-${HOME}/.config/zonos3-eval/secrets.env}"   # ZONOS3_GITHUB_TOKEN=<token>
UV_VERSION="0.9.17"
ZONOS3_PORT=1919   # serve.sh's port
# Server throughput settings (23 vCPUs on h200): text tokenizer processes beside the scheduler, the detokenizer, the API
# server and the client; prefill tokens per scheduler step, so a burst of requests is admitted in a few steps.
ZONOS3_TOKENIZERS="${ZONOS3_TOKENIZERS:-16}"
ZONOS3_PREFILL_TOKENS="${ZONOS3_PREFILL_TOKENS:-8192}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"
SUPPORTS_VOICE_CLONE=true
resolve_mode() { VOICE_CLONE=true voice_clone_mode; }   # speaker-embedding cloning is this backend's only mode

if [[ " ${STAGES:-generate} " == *" generate "* ]] && ! grep -q '^ZONOS3_GITHUB_TOKEN=.' "${ZONOS3_SECRETS_FILE}" 2>/dev/null; then
    echo "ERROR: ${ZONOS3_SECRETS_FILE} must hold ZONOS3_GITHUB_TOKEN=<token with read access to Zyphra/ZONOS3 and Zyphra/fusealign-multilingual>." >&2
    exit 1
fi

# ── Stage 1: install the pinned runtime, start its server, generate via the HTTP client. MODEL_CFG = "model_id max_running [lm]". ──
# max_running = the scheduler's running-request cap (and CUDA-graph batch); the client keeps 2x that many requests in
# flight per batch, so the scheduler's queue refills every slot as soon as a request finishes. lm = --model-path (empty:
# the release LM).
generate_stage() {
    local _ MAX_RUNNING SERVE_MODEL
    read -r _ MAX_RUNNING SERVE_MODEL <<< "${MODEL_CFG}"
    local CLIENT_BATCH=$((2 * MAX_RUNNING))
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN --secrets-file "${ZONOS3_SECRETS_FILE}" ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${ZONOS3_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            set -euo pipefail
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates ffmpeg build-essential > /dev/null
            curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
            git config --global url.\"https://x-access-token:\${ZONOS3_GITHUB_TOKEN}@github.com/\".insteadOf https://github.com/
            git clone -q ${ZONOS3_REPO} /opt/zonos3
            git -C /opt/zonos3 checkout -q ${ZONOS3_REV}
            git -C /opt/zonos3 submodule update --init -q third_party/fusealign-multilingual
            cd /opt/zonos3 && uv sync --locked --package zonos3-sgl --group phonemize
            SERVE_MODEL=${SERVE_MODEL} servers/sgl/scripts/serve.sh --max-running-requests ${MAX_RUNNING} --cuda-graph-max-bs ${MAX_RUNNING} \
                --max-prefill-length ${ZONOS3_PREFILL_TOKENS} --num-tokenizer ${ZONOS3_TOKENIZERS} > /tmp/zonos3_server.log 2>&1 &
            SERVER_PID=\$!
            trap 'rc=\$?; kill \$SERVER_PID 2>/dev/null || true; if [ \$rc -ne 0 ]; then tail -n 200 /tmp/zonos3_server.log; fi; exit \$rc' EXIT
            uv venv -q --python 3.12 /opt/client
            uv pip install -q --python /opt/client/bin/python 'datasets<4.0' 'huggingface_hub<1.0' soundfile librosa numpy tqdm num2words
            cd /app/zonos3 && /opt/client/bin/python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --language=${ASR_LANG} \
                --batch_size=${CLIENT_BATCH} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                --server_url=http://127.0.0.1:${ZONOS3_PORT} \
                --server_pid=\$SERVER_PID \
                ${VOICE_CLONE_FLAG}
            mkdir -p /results
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" ──────────────────────
# Seed-TTS + CV3 from DATASET_NAMESPACE's private copies; MiniMax: every language but uk (Qwen3-ASR has no
# Ukrainian) and yue (ZONOS3 has no Cantonese) — the fishaudio-drama-3 list.
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH}"
    "zero_shot zh zh ${CV3_EVAL_PATH}"
    "zero_shot fr fr ${CV3_EVAL_PATH}"
    "zero_shot ja ja ${CV3_EVAL_PATH}"
    "zero_shot ko ko ${CV3_EVAL_PATH}"
    "zero_shot de de ${CV3_EVAL_PATH}"
    "zero_shot es es ${CV3_EVAL_PATH}"
    "zero_shot it it ${CV3_EVAL_PATH}"
    "zero_shot ru ru ${CV3_EVAL_PATH}"
    "tts en en ${MINIMAX_EVAL_PATH}"
    "tts zh zh ${MINIMAX_EVAL_PATH}"
    "tts ja ja ${MINIMAX_EVAL_PATH}"
    "tts ko ko ${MINIMAX_EVAL_PATH}"
    "tts fr fr ${MINIMAX_EVAL_PATH}"
    "tts de de ${MINIMAX_EVAL_PATH}"
    "tts es es ${MINIMAX_EVAL_PATH}"
    "tts it it ${MINIMAX_EVAL_PATH}"
    "tts pt pt ${MINIMAX_EVAL_PATH}"
    "tts ru ru ${MINIMAX_EVAL_PATH}"
    "tts nl nl ${MINIMAX_EVAL_PATH}"
    "tts pl pl ${MINIMAX_EVAL_PATH}"
    "tts cs cs ${MINIMAX_EVAL_PATH}"
    "tts ro ro ${MINIMAX_EVAL_PATH}"
    "tts el el ${MINIMAX_EVAL_PATH}"
    "tts fi fi ${MINIMAX_EVAL_PATH}"
    "tts tr tr ${MINIMAX_EVAL_PATH}"
    "tts ar ar ${MINIMAX_EVAL_PATH}"
    "tts hi hi ${MINIMAX_EVAL_PATH}"
    "tts id id ${MINIMAX_EVAL_PATH}"
    "tts th th ${MINIMAX_EVAL_PATH}"
    "tts vi vi ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id max_running [lm]" (CLI args override); 512 = the largest cap measured (H100: RTFx 70-94) ──
MODEL_CONFIGS=(
    "Zyphra/ZONOS3-Mini 512 hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/rate30_20k_r2grpo"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
