#!/bin/bash
# SessionStart hook: install the momentum backtester's analysis stack so tests
# and the lab work in Claude Code on the web. Idempotent and non-interactive.
#
# Note: norgatedata (the live data adapter) is intentionally NOT installed -- it
# needs a running Norgate Data Updater + subscription, which an ephemeral remote
# container doesn't have. Everything except momentum_backtester_norgate.py runs
# without it.
set -euo pipefail

# Only run in the remote (web) environment; no-op on local machines.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

python3 -m pip install --quiet --disable-pip-version-check numpy pandas pytest

# Let pytest import the modules from the repo root.
echo 'export PYTHONPATH="."' >> "$CLAUDE_ENV_FILE"
