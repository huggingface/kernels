#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

cargo build -p hf-kernel-builder

"${REPO_ROOT}/target/debug/kernel-builder" generate-docs \
  | sed 's/hf-kernel-builder/kernel-builder/g' \
  | sed '1s/^# Command-Line Help for `kernel-builder`/# CLI reference for kernel-builder/' \
  | sed '/`--backends/,/^\*/{/^  Default value:/d;}' \
  > "${REPO_ROOT}/docs/source/builder-cli.md"

echo "Generated docs/source/builder-cli.md"
