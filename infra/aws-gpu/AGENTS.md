# AGENTS.md — driving the cuTile-rs AWS GPU smoke test end-to-end

**Audience:** an LLM coding agent (Claude Code, Cursor, Codex, etc.) helping a
contributor stand up a temporary NVIDIA GPU on AWS to validate a build of
[cutile-rs](https://github.com/NVlabs/cutile-rs).

Doubles as a checklist for a human walking through the same flow.

The companion [`README.md`](./README.md) documents the *Terraform* setup and
the file inventory. This file documents the *agent-driven workflow*: how to
onboard a contributor who has never touched AWS, how to handle credentials
safely, and what the agent must never do.

---

## Hard rules (non-negotiable)

These rules apply whether or not any tool-side enforcement is configured. If
the contributor is using Claude Code, the optional PreToolUse hook in
[`.guardrails/secret-guard.sh`](./.guardrails/secret-guard.sh) enforces most
of them mechanically — but defense in depth: obey them anyway.

1. **Never read secret files.** No `Read`, `cat`, `head`, `tail`, `less`,
   `bat`, `xxd`, `base64`, `cp`, `mv`, `tee` against:
   - `infra/aws-gpu/terraform.tfvars`
   - `infra/aws-gpu/.ssh/*.pem`
   - `~/.aws/credentials`
   - `~/.aws/config`
   - any `*.pem`

   To confirm a file exists, use `test -f <path> && echo exists`.

2. **Never put credentials on stdout.** Do not run `aws iam create-access-key`,
   `printenv AWS_*`, `env | grep AWS_*`, `echo $AWS_SECRET_ACCESS_KEY`, or
   `terraform output -raw <sensitive_value>` directly.

3. **Never set `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
   `AWS_SESSION_TOKEN` in subprocess env.** Pass `--profile <name>` to the AWS
   CLI; Terraform reads the named profile via `var.aws_profile`. The agent
   only ever knows the profile *name*, never the key material.

4. **Never ask the user to paste their access key or secret into the chat.**
   Have them run `aws configure --profile <name>` in their own terminal.

5. **Never run `terraform apply` or `up.sh` without explicit user approval.**
   `up.sh` runs `apply -auto-approve`, so getting the user's go-ahead first is
   the actual safety gate. Optionally offer `terraform plan -out=tfplan`
   beforehand for users new to Terraform — but don't make it a hard
   requirement, since experienced users will see it as friction.

6. **Always end with a teardown reminder.** A `g5.xlarge` runs ~$1/hr.
   Prompt the user to run `./down.sh` after success or failure.

---

## Persona triage — start here

Before doing anything, ask the user (one question) which case they're in:

| Case | Description | Skip to |
|------|-------------|---------|
| A | I have an AWS account, an IAM user with access keys, and `aws configure --profile <name>` already works. `aws sts get-caller-identity --profile <name>` returns my account. | [Configure terraform.tfvars](#configure-terraformtfvars) |
| B | I have an AWS account but no IAM user / access keys / AWS CLI yet. | [Persona B: Bootstrap an IAM user](#persona-b-bootstrap-an-iam-user) |
| C | I don't have an AWS account at all. | [Persona C: Create an AWS account](#persona-c-create-an-aws-account) |

---

## Persona C: Create an AWS account

The agent cannot do this — it requires a payment method and identity
verification. Tell the user, in order:

1. Go to <https://aws.amazon.com/> → **Create an AWS Account**.
2. Provide email, password, payment method, identity verification.
3. Sign in to the AWS Management Console as the **root user**.
4. **Immediately** harden the root account:
   - IAM → Users → root user → Security credentials → **Assign MFA device**.
   - Billing → Budgets → **Create budget** with a monthly cap (~$20) and an
     email alert at 80%. A forgotten `g5.xlarge` is the realistic cost risk
     here (~$24/day if left running).
5. Then proceed to **Persona B**.

---

## Persona B: Bootstrap an IAM user

The whole bootstrap is automated by [`bootstrap-iam.sh`](./bootstrap-iam.sh).
The only manual step is creating one short-lived access key in the AWS
Console — the chicken-and-egg problem (you can't call IAM without
already-authenticated credentials). The script then creates the long-lived
`cutile-runner` user and its access key, and prompts you to revoke the
bootstrap key.

### Why one manual step is unavoidable

You cannot call any IAM API without existing credentials. Terraform can't
solve this either — `aws_iam_access_key` writes the secret into Terraform
state, which is its own leakage problem. So someone has to make the first
credential. We make it short-lived, and the same script that uses it offers
to revoke it as its final step.

### Install the CLI tools

```bash
# macOS
brew install awscli terraform jq

# verify
aws --version
terraform -version
jq --version
```

Linux: use the system package manager or the official installers. The agent
should ask the user's distro rather than guessing.

### The flow

1. **In the AWS Console (one time, ~30 seconds):**
   Sign in as your root user. Top-right account menu → **Security
   credentials** → scroll to **Access keys** → **Create access key**. Accept
   the "this is your root user" warning checkbox. Copy the Access Key ID and
   Secret Access Key. **Do not paste them into the agent chat.**

2. **In your own terminal (NOT through the agent):**
   ```bash
   aws configure --profile bootstrap-admin
   # AWS Access Key ID:     <paste>
   # AWS Secret Access Key: <paste>
   # Default region:        us-east-1
   # Default output format: json
   ```

3. **Run the bootstrap script (in your own terminal, also NOT through the agent):**
   ```bash
   cd infra/aws-gpu
   ./bootstrap-iam.sh
   ```

4. **Confirm the revoke prompt:** type `y`. The script calls
   `aws iam delete-access-key`. The `bootstrap-admin` profile is now
   non-functional.

5. **Tell the agent "bootstrap done".** It will verify with
   `aws sts get-caller-identity --profile cutile` (returns Account/Arn/UserId,
   no secrets) and proceed to the GPU quota check.

### About the script

* `./bootstrap-iam.sh --help` shows all flags.
* **Idempotent** — re-running is safe; each step checks for existing state.
* `--dry-run` prints every API call it would make, without calling AWS.
* `--destroy` removes the policy and the `cutile-runner` user. Two modes:
  * **bootstrap-admin** (preferred): deletes everything cleanly if the
    bootstrap profile is still configured.
  * **self-destruct** (automatic fallback): uses `cutile-runner`'s own
    narrowly-scoped IAM perms. Detaches and deletes the policy and the
    access key, but the user object is left behind because AWS revokes auth
    the moment the key is deleted, before `DeleteUser` can run. The leftover
    user is harmless ($0/mo). To fully delete, re-create admin credentials
    and re-run `--destroy`.
* The least-privilege policy lives at
  [`iam-policy.json`](./iam-policy.json). It grants: EC2 lifecycle + describe
  + tags, security group + key pair management, SSM parameter read for the
  DLAMI lookup, service-quota read + request, STS GetCallerIdentity, and
  narrowly-scoped self-destruct (`DetachUserPolicy`, `DeleteAccessKey`,
  `DeleteUser`, `DeletePolicy`) limited to the `cutile-runner` user and the
  `cuTileTerraformLeastPrivilege` policy ARNs. No other IAM, no S3, no
  billing, no admin.

### Why the agent does not run the script

Two reasons:

1. The script captures the new IAM user's secret access key in shell
   variables and writes it to `~/.aws/credentials`. Even though it never
   echoes the secret, running it through the agent's Bash tool is unnecessary
   indirection — credentials should originate in your shell and stay there.
2. The optional `secret-guard` PreToolUse hook
   ([`.guardrails/secret-guard.sh`](./.guardrails/secret-guard.sh)) blocks
   `aws iam create-access-key` when invoked by the agent directly.

---

## Configure `terraform.tfvars`

Ask the user for:

- `aws_profile` — must match the profile name they used in `aws configure`.
- `aws_region` — default `us-east-1`.
- `instance_type` — default `g5.xlarge` (1× A10G, ~23 GB VRAM, ~$1/hr).

Write `infra/aws-gpu/terraform.tfvars` using your editor tool. Template:

```hcl
aws_region    = "us-east-1"
aws_profile   = "<profile-name-the-user-gave>"
instance_type = "g5.xlarge"
project_name  = "cutile-rs"
ssh_user      = "ubuntu"
```

The file is gitignored (see [`.gitignore`](./.gitignore)) so it will not be
committed. Do **not** read the file back to "verify" it. Confirm only that
it was written:

```bash
test -f infra/aws-gpu/terraform.tfvars && echo "tfvars written ($(wc -l < infra/aws-gpu/terraform.tfvars) lines)"
```

---

## Pre-apply: verify the GPU vCPU quota

Brand-new AWS accounts default to **0 vCPUs** for the *Running On-Demand G
and VT instances* quota (code `L-DB2E81BA`). A `g5.xlarge` is 4 vCPUs, so
`terraform apply` will fail with `VcpuLimitExceeded` if the quota hasn't been
raised. Check *before* applying:

```bash
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --region us-east-1 \
  --profile cutile \
  --query 'Quota.[Value,Adjustable]' --output text
```

Interpret:

| Output | Meaning | Action |
|--------|---------|--------|
| `4.0  True` (or higher) | Quota covers at least one g5.xlarge. | Proceed. |
| `0.0  True` | Quota is zero. New account. | Request an increase (below). |
| `<n>  False` | Non-adjustable in this account/region. | Switch region or contact AWS support. |

### Request a quota increase

```bash
aws service-quotas request-service-quota-increase \
  --service-code ec2 --quota-code L-DB2E81BA --desired-value 8 \
  --region us-east-1 --profile cutile \
  --query 'RequestedQuota.[Status,Id]' --output text
```

Status starts `PENDING`. Small increases on accounts with billing history
are often auto-approved within minutes; brand-new accounts go to manual
review (24–48 hr). Poll:

```bash
aws service-quotas list-requested-service-quota-change-history-by-quota \
  --service-code ec2 --quota-code L-DB2E81BA \
  --region us-east-1 --profile cutile \
  --query 'RequestedQuotas[0].[Status,DesiredValue,LastUpdatedAt]' --output text
```

When `Status` becomes `CASE_CLOSED` and `get-service-quota` returns the new
value, you're cleared.

---

## Stand up the GPU

```bash
cd infra/aws-gpu
./up.sh
```

Get explicit user approval first. `up.sh` runs `terraform init -upgrade` then
`terraform apply -auto-approve`. Takes ~2–3 minutes. When complete it prints:

```
GPU instance is up.
Public IP:  <ip>
SSH cmd:    ssh -i <path>.pem ubuntu@<ip>
```

The `.pem` file is at `infra/aws-gpu/.ssh/<key-name>.pem` (gitignored, mode
`0600`). The path itself is not secret — only the file's contents are.
`outputs.tf` marks `ssh_private_key_path` as `sensitive = true` so Terraform
redacts it in plan output.

### Optional: preview the plan first

```bash
cd infra/aws-gpu
terraform init -upgrade
terraform plan -out=tfplan
terraform show -no-color tfplan | grep -E '^Plan:|^\s+# ' | head -40
```

Typical output: `Plan: 5 to add, 0 to change, 0 to destroy` (EC2 instance,
security group, key pair, TLS private key, local pem file).

---

## Run the smoke test

```bash
cd infra/aws-gpu
./run-cutile-smoke.sh
```

This installs CUDA 13.2 / LLVM 21 / Rust on the host (~5 min, idempotent),
clones cutile-rs, and runs `cargo run -p cutile-examples --example add_refs`.

To test a fork or branch:

```bash
CUTILE_REPO="https://github.com/<user>/cutile-rs.git" \
CUTILE_REF="my-feature-branch" \
./run-cutile-smoke.sh
```

To run a different example:

```bash
CUTILE_EXAMPLE=saxpy ./run-cutile-smoke.sh
```

Report success/failure to the user. Expected runtime on first invocation is
~10 min (CUDA install + cargo build).

---

## Tear down — always offer

```bash
./down.sh
```

`down.sh` runs `terraform destroy`. **Always** offer this at the end of the
session, even on success. If the smoke test failed and the user wants to
investigate on the host, give them the SSH command and set an expectation:
*"Run `./down.sh` when you're finished — the instance bills hourly."*

---

## Optional: install the secret-guard hook (Claude Code only)

If the contributor is using Claude Code, the hook in
[`.guardrails/secret-guard.sh`](./.guardrails/secret-guard.sh) mechanically
denies the most common credential-leak patterns (e.g. `cat *.pem`,
`aws iam create-access-key`, `printenv AWS_*`). Wire it up by adding to
`.claude/settings.json` in the repo or user scope:

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Read", "hooks": [{ "type": "command", "command": "$CLAUDE_PROJECT_DIR/infra/aws-gpu/.guardrails/secret-guard.sh" }] },
      { "matcher": "Bash", "hooks": [{ "type": "command", "command": "$CLAUDE_PROJECT_DIR/infra/aws-gpu/.guardrails/secret-guard.sh" }] }
    ]
  }
}
```

The hook's behavior is pinned by 26 cases in
[`.guardrails/secret-guard.test.sh`](./.guardrails/secret-guard.test.sh).
Run them locally with:

```bash
bash infra/aws-gpu/.guardrails/secret-guard.test.sh
```

Other agent runtimes (Cursor, Codex, etc.) don't load this hook — so the
"Hard rules" above remain the only enforcement, and they're on you.

---

## Quick reference for the agent

```
Persona C (no AWS account)        → console signup → Persona B
Persona B (no IAM user / CLI)     → console IAM steps → aws configure (user) → verify
All personas                      → Write tfvars → optional plan → confirm → ./up.sh
                                  → ./run-cutile-smoke.sh → report → offer ./down.sh
```

When in doubt: do not read the file, do not echo the variable, ask the user.
