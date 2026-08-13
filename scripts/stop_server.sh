#!/bin/bash
#
# nohup으로 띄운 DentalLama API를 종료합니다.
#
#   ./scripts/stop_server.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
PID_FILE="${PROJECT_ROOT}/logs/server.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "[정보] PID 파일이 없습니다. 서버가 실행 중이지 않습니다."
  exit 0
fi

pid="$(cat "${PID_FILE}")"

if ! kill -0 "${pid}" 2>/dev/null; then
  echo "[정보] PID ${pid} 프로세스가 없습니다. 오래된 PID 파일을 삭제합니다."
  rm -f "${PID_FILE}"
  exit 0
fi

echo "[종료] PID ${pid}에 SIGTERM 전송"
kill "${pid}"

for _ in $(seq 1 20); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "[종료] 서버가 종료되었습니다."
    exit 0
  fi
  sleep 1
done

echo "[경고] 정상 종료되지 않아 SIGKILL을 전송합니다." >&2
kill -9 "${pid}" 2>/dev/null || true
rm -f "${PID_FILE}"
echo "[종료] 서버를 강제 종료했습니다."
