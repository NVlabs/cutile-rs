#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v terraform >/dev/null 2>&1; then
  echo "Terraform is required. Install Terraform and retry."
  exit 1
fi

terraform -chdir="${SCRIPT_DIR}" destroy -auto-approve "$@"
echo "Done. Terraform resources destroyed."
