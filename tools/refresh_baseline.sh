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
STAGED="$WORK/baseline.py"
trap 'rm -rf "$WORK"' EXIT

(cd "$WORK" && pip download "llmwhisperer-client==$VERSION" --no-deps -q && unzip -o -q ./*.whl -d x)

OUT="$REPO/tests/baseline/client_v2_$SLUG.py"
{
  echo "# Vendored from the released llmwhisperer-client $VERSION wheel on PyPI. DO NOT EDIT."
  echo "# Refresh with tools/refresh_baseline.sh when the parity baseline is intentionally moved."
  cat "$WORK/x/unstract/llmwhisperer/client_v2.py"
} > "$STAGED"

# Computed before the baseline is replaced: a digest step that fails afterwards
# leaves a moved baseline, a stale recorded digest and nothing to fix it with.
# sha256sum is GNU-only; shasum is what stock macOS ships.
if command -v sha256sum > /dev/null; then
  DIGEST="$(sha256sum "$STAGED" | cut -d' ' -f1)"
else
  DIGEST="$(shasum -a 256 "$STAGED" | cut -d' ' -f1)"
fi
mv "$STAGED" "$OUT"

echo "wrote $OUT"
echo "in tests/unit/compat_test.py set:"
echo "  BASELINE_VERSION = \"$VERSION\""
echo "  BASELINE_SHA256 = \"$DIGEST\""
