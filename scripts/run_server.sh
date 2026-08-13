#!/bin/bash
#
# DentalLama API를 포그라운드로 실행합니다.
# SSH 세션이 끊겨도 유지하려면 nohup으로 감싸서 실행하세요.
#
#   nohup ./scripts/run_server.sh >/dev/null 2>&1 &
#
# 요청/응답 로그: logs/server.log
# 종료: ./scripts/stop_server.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/server.log"
PID_FILE="${LOG_DIR}/server.pid"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/venv/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if kill -0 "${existing_pid}" 2>/dev/null; then
    echo "[오류] 서버가 이미 실행 중입니다. (PID ${existing_pid})" >&2
    echo "[오류] 종료하려면: ./scripts/stop_server.sh" >&2
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo $$ > "${PID_FILE}"

# 재실행 시 기존 로그를 덮어쓰고 logs/server.log에 stdout/stderr를 남깁니다.
exec >"${LOG_FILE}" 2>&1

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DentalLama API 시작"
echo "PID=$$  host=${HOST}  port=${PORT}  python=${PYTHON}"
echo "============================================================"

exec "${PYTHON}" -m uvicorn main:app --host "${HOST}" --port "${PORT}" --log-level info --proxy-headers --forwarded-allow-ips='*'
