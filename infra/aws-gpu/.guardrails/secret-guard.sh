#!/usr/bin/env bash
# PreToolUse hook: block the agent from leaking AWS / Terraform secrets into
# the transcript. Wired up in .claude/settings.json on the Read and Bash matchers.
#
# Reads a JSON envelope on stdin:
#   { "tool_name": "Read"|"Bash", "tool_input": { ... } }
# Emits a permissionDecision:"deny" response when a sensitive operation is detected.
# Otherwise exits 0 with no output (allow).

set -euo pipefail

input="$(cat)"
tool_name="$(printf '%s' "$input" | jq -r '.tool_name // ""')"

deny() {
  jq -nc --arg r "$1" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $r
    }
  }'
  exit 0
}

is_secret_path() {
  case "$1" in
    *terraform.tfvars|*terraform.tfvars.json) return 0 ;;
    *.pem) return 0 ;;
    */.aws/credentials|*/.aws/config) return 0 ;;
    "$HOME/.aws/credentials"|"$HOME/.aws/config") return 0 ;;
  esac
  return 1
}

case "$tool_name" in
  Read)
    fp="$(printf '%s' "$input" | jq -r '.tool_input.file_path // ""')"
    if is_secret_path "$fp"; then
      deny "secret-guard: refusing to Read $fp (AWS credentials, *.pem, or terraform.tfvars). If you need to confirm the file exists, use 'test -f' via Bash instead."
    fi
    ;;
  Bash)
    cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // ""')"

    view_tools='cat|head|tail|less|more|bat|xxd|od|strings|grep|awk|sed|base64|dd|cp|mv|tee'
    secret_re='(\.pem(\b|$)|terraform\.tfvars(\b|$)|\.aws/credentials(\b|$)|\.aws/config(\b|$))'

    # Split on shell separators (; | & and newlines) so that each subcommand is
    # evaluated independently. Without this, an innocuous pipeline like
    # `aws --version | head -1; test -f terraform.tfvars` gets flagged because
    # `head` and `terraform.tfvars` co-occur in the same string.
    chunks="$(printf '%s' "$cmd" | tr ';|&\n' '\n\n\n\n')"
    while IFS= read -r chunk; do
      [ -z "$chunk" ] && continue
      # Anchor the view-tool match to the start of the chunk (after optional
      # leading whitespace and optional sudo). This avoids false positives when
      # a view-tool token appears inside a quoted string passed to a different
      # command (e.g. `echo "use cat foo.pem to debug"`).
      if printf '%s' "$chunk" | grep -Eq "^[[:space:]]*(sudo[[:space:]]+)?($view_tools)([[:space:]]|$)" \
         && printf '%s' "$chunk" | grep -Eq "$secret_re"; then
        deny "secret-guard: subcommand '$chunk' appears to read or copy a secret file (*.pem, terraform.tfvars, ~/.aws/credentials, or ~/.aws/config). Refusing to put secret material on stdout."
      fi
    done <<EOF
$chunks
EOF

    if printf '%s' "$cmd" | grep -Eq '(^|[ |;&"`(])aws[ ]+iam[ ]+create-access-key'; then
      deny "secret-guard: 'aws iam create-access-key' prints the SecretAccessKey to stdout, which would land in the transcript. Have the user run this in their own terminal, or follow the bootstrap recipe in infra/aws-gpu/AGENTS.md that pipes the result straight to ~/.aws/credentials without echoing it."
    fi

    if printf '%s' "$cmd" | grep -Eq '(^|[ |;&"`(])(printenv|env)([ ]|$)' \
       && printf '%s' "$cmd" | grep -Eq 'AWS_(ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN)'; then
      deny "secret-guard: refusing to print AWS_* environment variables."
    fi

    if printf '%s' "$cmd" | grep -Eq 'echo[ ]+["'"'"']?\$\{?AWS_(ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN)'; then
      deny "secret-guard: refusing to echo AWS credential env vars."
    fi
    ;;
esac

exit 0
