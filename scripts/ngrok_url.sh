#!/bin/bash
#
# 실행 중인 ngrok 공개 URL을 출력합니다.
#
#   ./scripts/ngrok_url.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
URL_FILE="${PROJECT_ROOT}/logs/ngrok.url"
NGROK_API="${NGROK_API:-http://127.0.0.1:4040/api/tunnels}"

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

if [[ -z "${url}" && -f "${URL_FILE}" ]]; then
  url="$(cat "${URL_FILE}")"
fi

if [[ -z "${url}" ]]; then
  echo "[오류] ngrok 공개 URL을 찾지 못했습니다. ngrok이 실행 중인지 확인하세요." >&2
  echo "[오류] 시작: nohup ./scripts/run_ngrok.sh >/dev/null 2>&1 &" >&2
  exit 1
fi

echo "${url}"
echo "${url}" > "${URL_FILE}"
