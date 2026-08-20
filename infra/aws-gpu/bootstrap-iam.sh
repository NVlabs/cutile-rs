#!/usr/bin/env bash
# infra/aws-gpu/bootstrap-iam.sh
#
# Bootstrap an AWS IAM policy + user + access key for the cuTile GPU smoke
# test, using a one-time bootstrap profile that you delete immediately after.
#
# DESIGN NOTES — secret hygiene
#   * The new IAM user's access key never appears on stdout. The script
#     captures `aws iam create-access-key`'s JSON output into shell variables,
#     parses it with jq, writes the components to ~/.aws/credentials via
#     `aws configure set`, then unsets the variables.
#   * Access key IDs (the AKIA... part) ARE printed — those are public
#     identifiers, not secrets. Only the secret access key is sensitive.
#   * Idempotent: re-running is safe. Each step checks for existing state.
#   * Final step prompts whether to revoke the bootstrap key. Strongly
#     recommended — the bootstrap profile is typically your AWS root user.
#
# USAGE
#   ./bootstrap-iam.sh             create policy, user, access key, profile
#   ./bootstrap-iam.sh --dry-run   print actions, make no AWS API calls
#   ./bootstrap-iam.sh --destroy   detach + delete user + delete policy
#   ./bootstrap-iam.sh --help      show this header
#
# ENVIRONMENT OVERRIDES (defaults shown)
#   BOOTSTRAP_PROFILE=bootstrap-admin       AWS CLI profile with IAM perms
#   IAM_USER=cutile-runner                  new IAM user name
#   IAM_POLICY=cuTileTerraformLeastPrivilege   policy name
#   NEW_PROFILE=cutile                      AWS CLI profile name to write
#   AWS_REGION=us-east-1                    region for the new profile
#
# PRE-REQUISITES
#   * `aws` CLI v2 and `jq` installed.
#   * The bootstrap profile is configured (`aws configure --profile <name>`)
#     and has IAM permissions to create policies, users, access keys.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
POLICY_FILE="${SCRIPT_DIR}/iam-policy.json"

DRY_RUN=0
DESTROY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --destroy) DESTROY=1; shift ;;
    -h|--help)
      sed -n '1,40p' "$0" | sed -e 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

BOOTSTRAP_PROFILE="${BOOTSTRAP_PROFILE:-bootstrap-admin}"
IAM_USER="${IAM_USER:-cutile-runner}"
IAM_POLICY="${IAM_POLICY:-cuTileTerraformLeastPrivilege}"
NEW_PROFILE="${NEW_PROFILE:-cutile}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# ---------- helpers ----------
red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
say()    { printf '%s\n' "$*"; }

require() {
  command -v "$1" >/dev/null 2>&1 || { red "Missing dependency: $1"; exit 1; }
}

run_aws() {
  # Wrapper that no-ops in dry-run mode and prints the intended call instead.
  # Prints to stderr so callers that pipe stdout to /dev/null still surface it.
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '  [dry-run] aws %s\n' "$*" >&2
  else
    aws "$@"
  fi
}

# ---------- pre-flight ----------
require aws
require jq

if [[ ! -f "$POLICY_FILE" ]]; then
  red "Policy file not found: $POLICY_FILE"
  exit 1
fi
if ! jq -e '.Version and (.Statement|type=="array")' "$POLICY_FILE" >/dev/null; then
  red "Policy file is not a valid IAM policy: $POLICY_FILE"
  exit 1
fi

ACCOUNT_ID=""
CALLER_ARN=""
if [[ "$DRY_RUN" != 1 ]]; then
  if ! ID_JSON=$(aws sts get-caller-identity --profile "$BOOTSTRAP_PROFILE" --output json 2>&1); then
    red "Bootstrap profile '$BOOTSTRAP_PROFILE' is not configured or has invalid credentials."
    say "Configure it first:"
    say "  aws configure --profile $BOOTSTRAP_PROFILE"
    exit 1
  fi
  ACCOUNT_ID=$(echo "$ID_JSON" | jq -r '.Account')
  CALLER_ARN=$(echo "$ID_JSON" | jq -r '.Arn')
fi

POLICY_ARN="arn:aws:iam::${ACCOUNT_ID:-<account-id>}:policy/${IAM_POLICY}"

if [[ "$DRY_RUN" == 1 ]]; then
  yellow "=== DRY RUN — no AWS API calls will be made ==="
fi

say "Configuration:"
say "  bootstrap profile : $BOOTSTRAP_PROFILE"
say "  bootstrap caller  : ${CALLER_ARN:-<unknown in dry-run>}"
say "  account           : ${ACCOUNT_ID:-<unknown in dry-run>}"
say "  IAM user          : $IAM_USER"
say "  IAM policy        : $IAM_POLICY"
say "  new CLI profile   : $NEW_PROFILE"
say "  region            : $AWS_REGION"
say ""

# Detect root and warn (root user signing identity ends with :root)
if [[ "$CALLER_ARN" == *":root" ]]; then
  yellow "Note: bootstrap profile is using the AWS root user."
  yellow "      That is fine for a one-time bootstrap, but you MUST revoke this key after."
  say ""
fi

# ---------- destroy path ----------
if [[ "$DESTROY" == 1 ]]; then
  yellow "=== DESTROY: removing $IAM_USER and policy $IAM_POLICY ==="
  say ""

  # Pick which profile to use. Prefer BOOTSTRAP_PROFILE (full admin auth);
  # if that's been revoked, fall back to NEW_PROFILE (the cutile profile
  # itself, which has narrowly-scoped self-destruct IAM perms granted by
  # iam-policy.json). Self-destruct can detach + delete the policy and
  # access keys, but DeleteUser will fail because AWS revokes the cutile
  # profile's auth as soon as its access key is deleted — leaving an empty
  # user object behind ($0/mo, no perms, harmless).
  DESTROY_PROFILE="$BOOTSTRAP_PROFILE"
  DESTROY_MODE="bootstrap-admin"
  if [[ "$DRY_RUN" != 1 ]]; then
    if ! aws sts get-caller-identity --profile "$BOOTSTRAP_PROFILE" >/dev/null 2>&1; then
      yellow "Bootstrap profile [$BOOTSTRAP_PROFILE] is not configured or has been revoked."
      if aws sts get-caller-identity --profile "$NEW_PROFILE" >/dev/null 2>&1; then
        DESTROY_PROFILE="$NEW_PROFILE"
        DESTROY_MODE="self-destruct"
        yellow "Falling back to self-destruct via [$NEW_PROFILE]."
        yellow "  This will detach + delete the policy and the user's access key."
        yellow "  $IAM_USER itself will remain as an empty user object (no keys, no"
        yellow "  policies, \$0/mo, harmless). To fully delete, re-create admin"
        yellow "  credentials and re-run --destroy."
        say ""
      else
        red "Neither [$BOOTSTRAP_PROFILE] nor [$NEW_PROFILE] are usable. Cannot proceed."
        red "Configure one of them first (see 'aws configure --profile <name>')."
        exit 1
      fi
    fi
  fi

  # Order matters in self-destruct mode: do everything that needs auth
  # BEFORE deleting access keys (which kills our auth). So:
  #   1. Detach policy from user
  #   2. Delete policy versions + policy
  #   3. Delete access keys (cutile auth dies here)
  #   4. Try delete-user (succeeds in bootstrap-admin mode; fails cleanly in
  #      self-destruct mode and we report the leftover empty user object)

  # 1. Detach policy
  if [[ "$DRY_RUN" == 1 ]] || aws iam list-attached-user-policies --user-name "$IAM_USER" --profile "$DESTROY_PROFILE" \
       --query 'AttachedPolicies[?PolicyName==`'"$IAM_POLICY"'`].PolicyArn' --output text 2>/dev/null | grep -q .; then
    say "Detaching policy $IAM_POLICY from $IAM_USER"
    run_aws iam detach-user-policy --user-name "$IAM_USER" --policy-arn "$POLICY_ARN" --profile "$DESTROY_PROFILE"
  fi

  # 2. Delete policy (and any non-default versions)
  if [[ "$DRY_RUN" == 1 ]] || aws iam get-policy --policy-arn "$POLICY_ARN" --profile "$DESTROY_PROFILE" >/dev/null 2>&1; then
    VERSIONS=$(aws iam list-policy-versions --policy-arn "$POLICY_ARN" --profile "$DESTROY_PROFILE" \
                 --query 'Versions[?IsDefaultVersion==`false`].VersionId' --output text 2>/dev/null || true)
    for V in $VERSIONS; do
      [[ -z "$V" ]] && continue
      run_aws iam delete-policy-version --policy-arn "$POLICY_ARN" --version-id "$V" --profile "$DESTROY_PROFILE"
    done
    say "Deleting policy $IAM_POLICY"
    run_aws iam delete-policy --policy-arn "$POLICY_ARN" --profile "$DESTROY_PROFILE"
  fi

  # 3. Delete user's access keys (cutile auth dies here in self-destruct mode)
  if [[ "$DRY_RUN" == 1 ]]; then
    say "Deleting access key(s) for $IAM_USER"
    printf '  [dry-run] aws iam delete-access-key --user-name %s --access-key-id <each> --profile %s\n' "$IAM_USER" "$DESTROY_PROFILE" >&2
  else
    KEYS=$(aws iam list-access-keys --user-name "$IAM_USER" --profile "$DESTROY_PROFILE" \
             --query 'AccessKeyMetadata[].AccessKeyId' --output text 2>/dev/null || true)
    for KEY_ID in $KEYS; do
      [[ -z "$KEY_ID" ]] && continue
      say "Deleting access key $KEY_ID on $IAM_USER"
      aws iam delete-access-key --user-name "$IAM_USER" --access-key-id "$KEY_ID" --profile "$DESTROY_PROFILE"
    done
  fi

  # 4. Delete user — works with bootstrap-admin auth, fails cleanly in self-destruct
  if [[ "$DRY_RUN" == 1 ]]; then
    say "Deleting user $IAM_USER"
    run_aws iam delete-user --user-name "$IAM_USER" --profile "$DESTROY_PROFILE"
  elif aws iam get-user --user-name "$IAM_USER" --profile "$DESTROY_PROFILE" >/dev/null 2>&1; then
    say "Deleting user $IAM_USER"
    if aws iam delete-user --user-name "$IAM_USER" --profile "$DESTROY_PROFILE" 2>/dev/null; then
      green "  $IAM_USER deleted."
    else
      yellow "  delete-user failed (expected in self-destruct mode — auth was revoked"
      yellow "  when the access key was deleted)."
      yellow "  $IAM_USER remains as an empty user object (no keys, no policies)."
      yellow "  To fully delete: re-create admin credentials, re-run --destroy."
    fi
  fi

  green "Destroy complete."
  yellow "Optional: remove the [$NEW_PROFILE] (and dead [$BOOTSTRAP_PROFILE], if present)"
  yellow "          section from ~/.aws/credentials manually."
  exit 0
fi

# ---------- create path ----------

# 1. Policy
if [[ "$DRY_RUN" != 1 ]] && aws iam get-policy --policy-arn "$POLICY_ARN" --profile "$BOOTSTRAP_PROFILE" >/dev/null 2>&1; then
  say "Policy $IAM_POLICY already exists at $POLICY_ARN"
else
  say "Creating policy $IAM_POLICY from $POLICY_FILE"
  run_aws iam create-policy \
    --policy-name "$IAM_POLICY" \
    --policy-document "file://${POLICY_FILE}" \
    --description "Least-privilege policy for cutile-rs AWS GPU smoke test" \
    --profile "$BOOTSTRAP_PROFILE" >/dev/null
fi

# 2. User
if [[ "$DRY_RUN" != 1 ]] && aws iam get-user --user-name "$IAM_USER" --profile "$BOOTSTRAP_PROFILE" >/dev/null 2>&1; then
  say "User $IAM_USER already exists"
else
  say "Creating user $IAM_USER"
  run_aws iam create-user --user-name "$IAM_USER" --profile "$BOOTSTRAP_PROFILE" >/dev/null
fi

# 3. Attach policy
if [[ "$DRY_RUN" != 1 ]] && aws iam list-attached-user-policies --user-name "$IAM_USER" --profile "$BOOTSTRAP_PROFILE" \
       --query 'AttachedPolicies[?PolicyName==`'"$IAM_POLICY"'`]' --output text 2>/dev/null | grep -q .; then
  say "Policy $IAM_POLICY already attached to $IAM_USER"
else
  say "Attaching policy $IAM_POLICY to $IAM_USER"
  run_aws iam attach-user-policy --user-name "$IAM_USER" --policy-arn "$POLICY_ARN" --profile "$BOOTSTRAP_PROFILE"
fi

# 4. Access key + write to profile (the only step that handles secret material)
if [[ "$DRY_RUN" != 1 ]] && aws sts get-caller-identity --profile "$NEW_PROFILE" >/dev/null 2>&1; then
  yellow "Profile [$NEW_PROFILE] already works — skipping access key creation."
  say   "  To rotate, first delete the existing key:"
  say   "  aws iam list-access-keys --user-name $IAM_USER --profile $BOOTSTRAP_PROFILE"
  say   "  aws iam delete-access-key --user-name $IAM_USER --access-key-id <id> --profile $BOOTSTRAP_PROFILE"
else
  if [[ "$DRY_RUN" != 1 ]]; then
    EXISTING_KEYS=$(aws iam list-access-keys --user-name "$IAM_USER" --profile "$BOOTSTRAP_PROFILE" \
                      --query 'AccessKeyMetadata[].AccessKeyId' --output text 2>/dev/null | wc -w | tr -d ' ')
    if [[ "$EXISTING_KEYS" -ge 2 ]]; then
      red "User $IAM_USER already has 2 access keys (the AWS limit). Delete one first:"
      red "  aws iam list-access-keys --user-name $IAM_USER --profile $BOOTSTRAP_PROFILE"
      exit 1
    fi
  fi

  say "Creating access key for $IAM_USER and writing it to ~/.aws/credentials [$NEW_PROFILE]"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '  [dry-run] aws iam create-access-key --user-name %s --profile %s\n' "$IAM_USER" "$BOOTSTRAP_PROFILE"
    printf '  [dry-run] aws configure set aws_access_key_id <new-id>     --profile %s\n' "$NEW_PROFILE"
    printf '  [dry-run] aws configure set aws_secret_access_key <hidden> --profile %s\n' "$NEW_PROFILE"
    printf '  [dry-run] aws configure set region %s --profile %s\n' "$AWS_REGION" "$NEW_PROFILE"
    printf '  [dry-run] aws configure set output json --profile %s\n' "$NEW_PROFILE"
  else
    KEY_JSON=$(aws iam create-access-key --user-name "$IAM_USER" --profile "$BOOTSTRAP_PROFILE" --output json)
    NEW_KEY_ID=$(echo "$KEY_JSON" | jq -r '.AccessKey.AccessKeyId')
    NEW_SECRET=$(echo "$KEY_JSON" | jq -r '.AccessKey.SecretAccessKey')
    KEY_JSON=""

    aws configure set aws_access_key_id     "$NEW_KEY_ID" --profile "$NEW_PROFILE"
    aws configure set aws_secret_access_key "$NEW_SECRET" --profile "$NEW_PROFILE"
    aws configure set region                "$AWS_REGION" --profile "$NEW_PROFILE"
    aws configure set output                json          --profile "$NEW_PROFILE"
    NEW_SECRET=""

    say "  Wrote access key $NEW_KEY_ID to profile [$NEW_PROFILE] (secret not shown)"
  fi
fi

# 5. Verify the new profile works (no secret material in this output).
#    AWS IAM is eventually consistent: a brand-new access key can take 5-30
#    seconds before STS accepts it. Retry on InvalidClientTokenId rather than
#    aborting (which would skip the revoke prompt and leave the bootstrap key
#    active longer than necessary).
if [[ "$DRY_RUN" != 1 ]]; then
  say ""
  say "Verifying new profile [$NEW_PROFILE] (allowing time for AWS eventual consistency)..."
  VERIFY_OK=0
  for attempt in 1 2 3 4 5 6 7 8; do
    if VERIFY_OUT=$(aws sts get-caller-identity --profile "$NEW_PROFILE" 2>&1); then
      VERIFY_OK=1
      printf '%s\n' "$VERIFY_OUT"
      break
    fi
    say "  attempt $attempt/8: not yet propagated, sleeping 5s..."
    sleep 5
  done
  if [[ "$VERIFY_OK" != 1 ]]; then
    yellow ""
    yellow "Verification did not succeed within ~40 seconds."
    yellow "Most likely AWS eventual consistency — retry manually after another minute:"
    yellow "  aws sts get-caller-identity --profile $NEW_PROFILE"
    yellow "Continuing to the revoke prompt regardless (the bootstrap key should not"
    yellow "stay around longer than necessary)."
  fi
fi

green ""
green "Bootstrap complete."
green "  Policy:   $IAM_POLICY ($POLICY_ARN)"
green "  User:     $IAM_USER"
green "  Profile:  $NEW_PROFILE   (try: aws sts get-caller-identity --profile $NEW_PROFILE)"
green ""

# 6. Offer to revoke the bootstrap key
if [[ "$DRY_RUN" == 1 ]]; then
  yellow "[dry-run] Would now offer to revoke the bootstrap access key on profile [$BOOTSTRAP_PROFILE]."
  exit 0
fi

BOOTSTRAP_KEY_ID="$(aws configure get aws_access_key_id --profile "$BOOTSTRAP_PROFILE" 2>/dev/null || true)"
if [[ -z "$BOOTSTRAP_KEY_ID" ]]; then
  yellow "Could not read the bootstrap access key id from ~/.aws/credentials. Skipping revoke prompt."
  yellow "Revoke the bootstrap key manually in the AWS Console at your earliest convenience."
  exit 0
fi

yellow "===== Revoke the bootstrap access key ====="
yellow "The bootstrap profile [$BOOTSTRAP_PROFILE] is using access key: $BOOTSTRAP_KEY_ID"
if [[ "$CALLER_ARN" == *":root" ]]; then
  yellow "This is your AWS ROOT USER access key. Revoking it now is strongly recommended."
fi
yellow ""
read -r -p "Revoke this access key now? [y/N] " ans
case "$ans" in
  [Yy]|[Yy][Ee][Ss])
    aws iam delete-access-key --access-key-id "$BOOTSTRAP_KEY_ID" --profile "$BOOTSTRAP_PROFILE"
    green "Revoked access key $BOOTSTRAP_KEY_ID."
    yellow "The bootstrap profile [$BOOTSTRAP_PROFILE] is now non-functional."
    yellow "Optional: remove the [$BOOTSTRAP_PROFILE] section from ~/.aws/credentials manually."
    ;;
  *)
    yellow "Skipped. Revoke manually:"
    yellow "  Console: https://console.aws.amazon.com/iam/home#/security_credentials"
    yellow "  CLI:     aws iam delete-access-key --access-key-id $BOOTSTRAP_KEY_ID --profile $BOOTSTRAP_PROFILE"
    ;;
esac
