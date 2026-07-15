#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
world="$repository_root/simulation/worlds/robot_arm_pick_and_place.wbt"

if [[ -n "${WEBOTS_HOME:-}" && -x "$WEBOTS_HOME/webots" ]]; then
  webots_binary="$WEBOTS_HOME/webots"
elif command -v webots >/dev/null 2>&1; then
  webots_binary="$(command -v webots)"
else
  echo "Webots was not found. Set WEBOTS_HOME or add webots to PATH." >&2
  exit 1
fi

exec "$webots_binary" "$@" "$world"
