#!/bin/bash
#
# 로컬 API(기본 8001)를 ngrok으로 외부에 노출합니다.
# SSH 세션이 끊겨도 유지하려면 nohup으로 감싸서 실행하세요.
#
#   nohup ./scripts/run_ngrok.sh >/dev/null 2>&1 &
#
# 로그: logs/ngrok.log
# 공개 URL: logs/ngrok.url  또는  ./scripts/ngrok_url.sh
# 종료: ./scripts/stop_ngrok.sh
#
# 고정 도메인이 있으면:
#   NGROK_DOMAIN=your-name.ngrok-free.app nohup ./scripts/run_ngrok.sh >/dev/null 2>&1 &
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/ngrok.log"
PID_FILE="${LOG_DIR}/ngrok.pid"
URL_FILE="${LOG_DIR}/ngrok.url"
PORT="${PORT:-8001}"
NGROK_BIN="${NGROK_BIN:-ngrok}"
NGROK_API="${NGROK_API:-http://127.0.0.1:4040/api/tunnels}"

if ! command -v "${NGROK_BIN}" >/dev/null 2>&1; then
  echo "[오류] ngrok을 찾을 수 없습니다. ngrok이 PATH에 있는지 확인하세요." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if kill -0 "${existing_pid}" 2>/dev/null; then
    echo "[오류] ngrok이 이미 실행 중입니다. (PID ${existing_pid})" >&2
    echo "[오류] 종료하려면: ./scripts/stop_ngrok.sh" >&2
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

ngrok_pid=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "${ngrok_pid}" ]] && kill -0 "${ngrok_pid}" 2>/dev/null; then
    kill "${ngrok_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

write_public_url() {
  local url=""
  for _ in $(seq 1 30); do
    sleep 1
    url="$(
      curl -sf "${NGROK_API}" 2>/dev/null | python -c '
import json
import sys

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

tunnels = data.get("tunnels") or []
https_urls = [
    t.get("public_url") for t in tunnels
    if (t.get("public_url") or "").startswith("https://")
]
if https_urls:
    print(https_urls[0])
elif tunnels and tunnels[0].get("public_url"):
    print(tunnels[0]["public_url"])
' || true
    )"
    if [[ -n "${url}" ]]; then
      echo "${url}" > "${URL_FILE}"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] public url: ${url}"
      return 0
    fi
  done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] public URL을 아직 확인하지 못했습니다. ./scripts/ngrok_url.sh 로 다시 확인하세요."
}

# 재실행 시 기존 로그를 덮어쓰고 logs/ngrok.log에 stdout/stderr를 남깁니다.
exec >"${LOG_FILE}" 2>&1

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ngrok 시작  port=${PORT}"
echo "============================================================"

ngrok_args=(http "${PORT}" --log=stdout --log-format=term --log-level=info)
if [[ -n "${NGROK_DOMAIN:-}" ]]; then
  ngrok_args+=(--url="${NGROK_DOMAIN}")
  echo "domain=${NGROK_DOMAIN}"
fi

"${NGROK_BIN}" "${ngrok_args[@]}" &
ngrok_pid=$!
echo "${ngrok_pid}" > "${PID_FILE}"

write_public_url &
set +e
wait "${ngrok_pid}"
ngrok_status=$?
set -e

trap - EXIT INT TERM
rm -f "${PID_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ngrok 종료 (exit ${ngrok_status})"
exit "${ngrok_status}"
