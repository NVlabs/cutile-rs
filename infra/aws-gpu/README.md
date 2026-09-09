# `infra/aws-gpu` — Temporary AWS GPU host for cuTile smoke tests

Provisions a short-lived `g5.xlarge` (NVIDIA A10G, sm_86), installs CUDA 13.2
+ LLVM 21 + Rust on it, clones cutile-rs, and runs a known-good example
(`cutile-examples --example add_refs`). For contributors who don't already
own a CUDA 13.2 + sm_80+ workstation. Not part of the regular build —
`scripts/run_all.sh` does not touch this folder.

## Quick start

### 1. Get the code

```bash
git clone https://github.com/NVlabs/cutile-rs.git
cd cutile-rs/infra/aws-gpu
```

### 2. Install local CLI deps (one time)

```bash
# macOS
brew install awscli terraform jq
# verify
aws --version && terraform -version && jq --version
```

Linux: use your package manager.

### 3. AWS credentials

Pick the path that matches you.

**Already have a working AWS profile** (`aws sts get-caller-identity --profile <name>` returns your account):
→ skip to step 4.

**Have an AWS account but no IAM user / access keys yet:**

```bash
# 3a. AWS Console (root user) → Security credentials → Create access key.
#     Copy the AKIA... and the secret. Don't paste them anywhere yet.

# 3b. In your terminal:
aws configure --profile bootstrap-admin
#   AWS Access Key ID:     <paste>
#   AWS Secret Access Key: <paste>
#   Default region:        us-east-1

# 3c. Run the bootstrap. Creates a least-privilege "cutile" profile,
#     then prompts you to revoke the bootstrap key. Type 'y' at the prompt.
./bootstrap-iam.sh
```

**No AWS account at all:** sign up at <https://aws.amazon.com/>, set up MFA
and a billing budget, then come back to this step. `AGENTS.md` walks through
this in detail.

### 4. Configure `terraform.tfvars`

```bash
cp terraform.tfvars.example terraform.tfvars
# edit aws_profile to match the name you used in `aws configure --profile <name>`
# (e.g. "cutile" if you ran bootstrap-iam.sh)
```

### 5. Check the GPU vCPU quota

Skip if you've launched a `g5` in this region before.

```bash
aws service-quotas get-service-quota \
  --service-code ec2 --quota-code L-DB2E81BA \
  --region us-east-1 --profile cutile \
  --query 'Quota.[Value,Adjustable]' --output text
```

Need ≥ `4.0`. If `0.0`, request 8:

```bash
aws service-quotas request-service-quota-increase \
  --service-code ec2 --quota-code L-DB2E81BA --desired-value 8 \
  --region us-east-1 --profile cutile
```

Wait for AWS approval (often minutes; brand-new accounts can take 24–48 hr).

### 6. Stand up, smoke, tear down

```bash
./up.sh                  # ~3 min (terraform apply)
./run-cutile-smoke.sh    # ~10 min first run (CUDA + LLVM + cargo build)
./down.sh                # ALWAYS run when done — g5.xlarge bills ~$1/hr
```

Success looks like Cargo's `Running target/debug/examples/add_refs` near the
end of the output, exit 0.

To test a fork or branch:

```bash
CUTILE_REPO="https://github.com/<you>/cutile-rs.git" \
CUTILE_REF="my-branch" \
./run-cutile-smoke.sh
```

## VS Code Remote-SSH (optional)

Add to `~/.ssh/config`:

```sshconfig
Host cutile-gpu
    HostName <public-ip-from-terraform-output>
    User ubuntu
    IdentityFile <repo>/infra/aws-gpu/.ssh/<run-id>.pem
    StrictHostKeyChecking accept-new
```

Then **Remote-SSH: Connect to Host…** → `cutile-gpu`. The cutile-rs checkout
is at `/home/ubuntu/cutile-rs`.

## Files

| File | Purpose |
|------|---------|
| `main.tf`, `variables.tf`, `outputs.tf`, `versions.tf` | Terraform definition. |
| `terraform.tfvars.example` | Copy to `terraform.tfvars` and edit. Real `.tfvars` is gitignored. |
| `up.sh` / `down.sh` | `terraform apply` / `destroy` wrappers. |
| `ssh.sh` | SSH into the instance with auto-detected user + key. |
| `userdata.sh` | Cloud-init: minimal apt deps. |
| `run-cutile-smoke.sh` | Install CUDA/LLVM/Rust, clone cutile-rs, run an example. |
| `bootstrap-iam.sh` + `iam-policy.json` | One-shot least-privilege IAM user bootstrap (optional). |
| `AGENTS.md` | LLM-agent runbook for driving this flow safely. |
| `.guardrails/` | Optional Claude Code PreToolUse hook + tests blocking secret-leak commands. |
| `.ssh/` | Generated `.pem` files — created by Terraform, gitignored. |
