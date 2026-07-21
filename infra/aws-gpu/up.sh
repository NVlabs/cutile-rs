#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v terraform >/dev/null 2>&1; then
  echo "Terraform is required. Install Terraform and retry."
  exit 1
fi

terraform -chdir="${SCRIPT_DIR}" init -upgrade
terraform -chdir="${SCRIPT_DIR}" apply -auto-approve "$@"

echo
echo "GPU instance is up."
echo "Public IP:  $(terraform -chdir="${SCRIPT_DIR}" output -raw public_ip)"
echo "SSH cmd:    $(terraform -chdir="${SCRIPT_DIR}" output -raw ssh_command)"
echo
echo "Next:"
echo "  ${SCRIPT_DIR}/run-cutile-smoke.sh"
echo
echo "When done (to avoid charges):"
echo "  ${SCRIPT_DIR}/down.sh"
