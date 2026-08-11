#!/usr/bin/env bash
# Refresh the vendored parity baseline in tests/baseline/.
#
# The compat suite compares this client against a fixed published one, not
# against the working tree — a baseline that moves with local edits measures
# nothing. It is vendored rather than resolved at test time so the suite stays
# offline, and refreshing it is a deliberate act with a reviewable diff.
#
#   ./tools/refresh_baseline.sh 0e9fda3 pr34
set -euo pipefail

REF="${1:?usage: refresh_baseline.sh <git-ref> <slug>}"
SLUG="${2:?usage: refresh_baseline.sh <git-ref> <slug>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/tests/baseline/client_v2_$SLUG.py"

{
  echo "# Vendored from llm-whisperer-python-client $REF, the parity baseline."
  echo "# DO NOT EDIT. Refresh with tools/refresh_baseline.sh when the baseline moves."
  git -C "$REPO" show "$REF:src/unstract/llmwhisperer/client_v2.py"
} > "$OUT"

echo "wrote $OUT"
echo "update BASELINE_REF in tests/test_compat.py to match"
