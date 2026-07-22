#!/usr/bin/env bash
# Deploy endpoint.gs to the Apps Script project named by
# AUTOLUME_CRASH_SCRIPT_ID in the repo-root .env. Pushes the code, cuts a
# version and updates the web app deployment (created on first run), then
# writes the deployment URL and a generated token into .env. The Script
# Property AUTOLUME_TOKEN and a one-time setup() run stay manual, in the
# editor (see README.md).
#
# One-time setup:
#   1. Enable the Apps Script API: https://script.google.com/home/usersettings
#   2. npx --yes @google/clasp login
set -euo pipefail
cd "$(dirname "$0")"

ENV_FILE="../../.env"
[[ -f "$ENV_FILE" ]] || { echo "error: repo-root .env missing; see .env.example" >&2; exit 1; }
get() {
  grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- \
    | sed -E 's/[[:space:]]+#.*$//; s/^[[:space:]]+//; s/[[:space:]]+$//' | tr -d "'\""
}
set_env() {
  local key="$1" value="$2" tmp
  if grep -qE "^$key=" "$ENV_FILE"; then
    tmp="$(mktemp)"
    awk -v k="$key" -v v="$value" 'index($0, k "=") == 1 { $0 = k "=" v } { print }' \
      "$ENV_FILE" > "$tmp" && mv "$tmp" "$ENV_FILE"
  else
    printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
  fi
}
SCRIPT_ID="$(get AUTOLUME_CRASH_SCRIPT_ID)"
[[ -n "$SCRIPT_ID" ]] || { echo "error: AUTOLUME_CRASH_SCRIPT_ID missing in .env" >&2; exit 1; }

# The project context lives in a throwaway directory so a stale .clasp.json
# can never target an outdated script id.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
printf '{"scriptId":"%s","rootDir":"%s"}\n' "$SCRIPT_ID" "$PWD" > "$TMP/.clasp.json"
clasp() { npx --yes @google/clasp -P "$TMP/.clasp.json" "$@"; }

clasp push --force
DESCRIPTION="deploy.sh $(date +%Y-%m-%d) $(git rev-parse --short HEAD 2>/dev/null || echo untracked)"
VERSION="$(clasp create-version "$DESCRIPTION" | grep -oE '[0-9]+' | tail -1)"
[[ -n "$VERSION" ]] || { echo "error: could not parse version from clasp create-version" >&2; exit 1; }

DEPLOYMENT_ID="$(clasp list-deployments "$SCRIPT_ID" | grep -v '@HEAD' \
  | grep -oE 'AKfycb[A-Za-z0-9_-]+' | head -1 || true)"
if [[ -n "$DEPLOYMENT_ID" ]]; then
  clasp update-deployment -V "$VERSION" -d "$DESCRIPTION" "$DEPLOYMENT_ID" > /dev/null
else
  DEPLOYMENT_ID="$(clasp create-deployment -V "$VERSION" -d "$DESCRIPTION" \
    | grep -oE 'AKfycb[A-Za-z0-9_-]+' | head -1)"
fi
[[ -n "$DEPLOYMENT_ID" ]] || { echo "error: could not resolve a deployment id" >&2; exit 1; }

URL="https://script.google.com/macros/s/$DEPLOYMENT_ID/exec"
echo "Deployed version $VERSION"
echo "Web app URL: $URL"
if [[ "$(get AUTOLUME_CRASH_REPORT_URL)" != "$URL" ]]; then
  set_env AUTOLUME_CRASH_REPORT_URL "$URL"
  echo "Wrote AUTOLUME_CRASH_REPORT_URL to .env"
fi

TOKEN="$(get AUTOLUME_CRASH_REPORT_TOKEN)"
if [[ -z "$TOKEN" ]]; then
  TOKEN="$(openssl rand -hex 32)"
  set_env AUTOLUME_CRASH_REPORT_TOKEN "$TOKEN"
  echo "Generated AUTOLUME_CRASH_REPORT_TOKEN and wrote it to .env"
  echo
  echo "Finish in the editor (script.google.com/home/projects/$SCRIPT_ID/edit):"
  echo "  1. Run the setup function once and accept the authorization prompt"
  echo "  2. Project Settings > Script Properties: set AUTOLUME_TOKEN to:"
  echo "     $TOKEN"
fi
