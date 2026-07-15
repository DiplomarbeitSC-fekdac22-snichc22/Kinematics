#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KINEMATICS_WEBOTS_QUIT_WHEN_DONE=1

"$repository_root/simulation/scripts/run_webots.sh" \
  --stdout \
  --stderr \
  --batch \
  --mode=fast
