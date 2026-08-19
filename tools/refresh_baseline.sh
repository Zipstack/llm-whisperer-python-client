#!/usr/bin/env bash
# Refresh the vendored parity baseline in tests/baseline/.
#
# The compat suite compares this client against the last RELEASED one, not
# against the working tree — a baseline that moves with local edits measures
# nothing. It is taken from the published wheel rather than from a git ref
# because the wheel is what callers actually have installed, and it is vendored
# rather than downloaded at test time so the suite stays offline.
#
#   ./tools/refresh_baseline.sh 2.8.1
set -euo pipefail

VERSION="${1:?usage: refresh_baseline.sh <released-version>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLUG="${VERSION//./_}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

(cd "$WORK" && pip download "llmwhisperer-client==$VERSION" --no-deps -q && unzip -o -q ./*.whl -d x)

OUT="$REPO/tests/baseline/client_v2_$SLUG.py"
{
  echo "# Vendored from the released llmwhisperer-client $VERSION wheel on PyPI. DO NOT EDIT."
  echo "# Refresh with tools/refresh_baseline.sh when the parity baseline is intentionally moved."
  cat "$WORK/x/unstract/llmwhisperer/client_v2.py"
} > "$OUT"

echo "wrote $OUT"
echo "in tests/unit/compat_test.py set:"
echo "  BASELINE_VERSION = \"$VERSION\""
echo "  BASELINE_SHA256 = \"$(sha256sum "$OUT" | cut -d' ' -f1)\""
