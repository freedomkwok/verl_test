#!/usr/bin/env bash
set -euo pipefail

# Get the value (prefer direct env var; fall back to parsing `env` output)
port="${VAST_TCP_PORT_5678:-}"
if [[ -z "${port}" ]]; then
  port="$(env | sed -n 's/^VAST_TCP_PORT_5678=\(.*\)$/\1/p')"
fi

if [[ -z "${port}" ]]; then
  echo "VAST_TCP_PORT_5678 is not set." >&2
  exit 1
fi

# Copy to clipboard (macOS, Wayland, X11, or WSL/Windows)
if command -v pbcopy >/dev/null 2>&1; then
  printf %s "$port" | pbcopy
elif command -v wl-copy >/dev/null 2>&1; then
  printf %s "$port" | wl-copy
elif command -v xclip >/dev/null 2>&1; then
  printf %s "$port" | xclip -selection clipboard
elif command -v clip.exe >/dev/null 2>&1; then
  printf %s "$port" | clip.exe
else
  echo "Copied value: $port"
  echo "(No clipboard tool found. Install one of: pbcopy, wl-copy, xclip, or use clip.exe on WSL.)"
  exit 0
fi

echo "Copied to clipboard: $port"
