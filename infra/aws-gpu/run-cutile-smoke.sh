#!/usr/bin/env bash
# Smoke test: SSH into the Terraform-provisioned EC2 GPU host, install build
# prerequisites (idempotent), clone cutile-rs from upstream, and run a known-
# good GPU example (`cutile-examples --example add_refs`). Prints nvidia-smi
# and tileiras versions as part of the run so failures are easy to triage.
#
# Inputs (from `terraform output` in this folder):
#   public_ip, ssh_private_key_path, remote_workdir, ssh_user
#
# Secrets handling:
#   * Never echoes the .pem path's contents — only its *path* is referenced.
#   * Does not export AWS_* env vars; relies on the configured AWS profile.
#   * The remote does not need AWS credentials at all (it just builds Rust).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for cmd in terraform ssh; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd is required."; exit 1; }
done

CUTILE_REPO="${CUTILE_REPO:-https://github.com/NVlabs/cutile-rs.git}"
CUTILE_REF="${CUTILE_REF:-main}"
CUTILE_EXAMPLE="${CUTILE_EXAMPLE:-add_refs}"

PUBLIC_IP="$(terraform -chdir="${SCRIPT_DIR}" output -raw public_ip)"
SSH_KEY_PATH="$(terraform -chdir="${SCRIPT_DIR}" output -raw ssh_private_key_path)"
REMOTE_DIR="$(terraform -chdir="${SCRIPT_DIR}" output -raw remote_workdir)"
PREFERRED_SSH_USER="$(terraform -chdir="${SCRIPT_DIR}" output -raw ssh_user 2>/dev/null || echo "ubuntu")"

SSH_OPTS=(-i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)

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
  echo "Unable to authenticate via SSH with any known user."
  echo "Tried: ${PREFERRED_SSH_USER}, ubuntu, ec2-user, admin"
  echo "Check key path, instance state, security group ingress CIDR, and AMI SSH user."
  exit 1
fi
echo "Using SSH user: ${SSH_USER}"
echo "Public IP:      ${PUBLIC_IP}"
echo "Remote workdir: ${REMOTE_DIR}"

echo "Installing CUDA 13.2 / LLVM 21 / Rust on EC2 (idempotent)..."
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${PUBLIC_IP}" \
    "REMOTE_DIR='${REMOTE_DIR}' CUTILE_REPO='${CUTILE_REPO}' CUTILE_REF='${CUTILE_REF}' bash -s" <<'REMOTE'
set -euo pipefail

sudo apt-get update -qq
sudo apt-get install -y -qq build-essential git curl wget jq pkg-config ca-certificates unzip

# Rust (rustup) — cuTile needs 1.89+ per repo README.
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
rustup toolchain install stable --profile minimal
rustup default stable

# LLVM 21 (cuda-tile uses MLIR from LLVM 21).
if ! command -v llvm-config >/dev/null 2>&1 || [[ "$(llvm-config --version | cut -d. -f1)" != "21" ]]; then
  wget -q https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh
  chmod +x /tmp/llvm.sh
  sudo /tmp/llvm.sh 21
  sudo apt-get install -y -qq libmlir-21-dev mlir-21-tools
  sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/lib/llvm-21/bin/llvm-config 1
fi

# CUDA 13.2 toolkit (the DLAMI base ships only the driver).
if ! command -v nvcc >/dev/null 2>&1 || ! nvcc --version | grep -q "release 13.2"; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update -qq
  sudo apt-get install -y -qq cuda-toolkit-13-2
fi

# Persist env for future logins (idempotent).
for line in \
  'export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.2' \
  'export CUDA_TILE_USE_LLVM_INSTALL_DIR=/usr/lib/llvm-21' \
  'export PATH=/usr/local/cuda-13.2/bin:$PATH'; do
  grep -qF "$line" "$HOME/.bashrc" || echo "$line" >> "$HOME/.bashrc"
done

# Clone or update cutile-rs.
mkdir -p "$(dirname "$REMOTE_DIR")"
if [[ ! -d "$REMOTE_DIR/.git" ]]; then
  git clone "$CUTILE_REPO" "$REMOTE_DIR"
fi
git -C "$REMOTE_DIR" fetch --quiet origin
git -C "$REMOTE_DIR" checkout "$CUTILE_REF"
git -C "$REMOTE_DIR" pull --ff-only --quiet || true
REMOTE

echo "Running cutile-rs smoke example on EC2 (cutile-examples --example ${CUTILE_EXAMPLE})..."
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${PUBLIC_IP}" \
    "REMOTE_DIR='${REMOTE_DIR}' CUTILE_EXAMPLE='${CUTILE_EXAMPLE}' bash -s" <<'REMOTE'
set -euo pipefail
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.2
export CUDA_TILE_USE_LLVM_INSTALL_DIR=/usr/lib/llvm-21
export PATH=/usr/local/cuda-13.2/bin:$PATH

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi missing. The DLAMI should include the NVIDIA driver." >&2
  exit 1
fi
if ! command -v tileiras >/dev/null 2>&1; then
  echo "tileiras not on PATH. Expected in /usr/local/cuda-13.2/bin (provided by cuda-toolkit-13-2)." >&2
  exit 1
fi

nvidia-smi
nvcc --version | head -n 1
llvm-config --version
tileiras --version || true

cd "$REMOTE_DIR"
cargo run -p cutile-examples --example "$CUTILE_EXAMPLE"
REMOTE

echo
echo "Smoke test complete. Tear down with ./down.sh when finished (g5.xlarge bills hourly)."
