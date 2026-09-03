#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v terraform >/dev/null 2>&1; then
  echo "Terraform is required. Install Terraform and retry."
  exit 1
fi

SSH_KEY_PATH="$(terraform -chdir="${SCRIPT_DIR}" output -raw ssh_private_key_path)"
PUBLIC_IP="$(terraform -chdir="${SCRIPT_DIR}" output -raw public_ip)"
PREFERRED_SSH_USER="$(terraform -chdir="${SCRIPT_DIR}" output -raw ssh_user 2>/dev/null || echo "ubuntu")"

SSH_OPTS=(-i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=accept-new)

resolve_ssh_user() {
  local -a candidates=("${PREFERRED_SSH_USER}" "ubuntu" "ec2-user" "admin")
  local user
  for user in "${candidates[@]}"; do
    if ssh "${SSH_OPTS[@]}" -o BatchMode=yes -o ConnectTimeout=8 "${user}@${PUBLIC_IP}" "true" >/dev/null 2>&1; then
      echo "${user}"
      return 0
    fi
  done
  return 1
}

if ! SSH_USER="$(resolve_ssh_user)"; then
  echo "Unable to authenticate via SSH with users: ${PREFERRED_SSH_USER}, ubuntu, ec2-user, admin"
  exit 1
fi

exec ssh "${SSH_OPTS[@]}" "${SSH_USER}@${PUBLIC_IP}"
