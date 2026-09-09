#!/usr/bin/env bash
# Test harness for .claude/hooks/secret-guard.sh
#
# WHY THIS FILE EXISTS
#   secret-guard.sh blocks Claude Code from leaking AWS / Terraform secrets
#   into the transcript. The regex-based heuristics inside it are easy to
#   accidentally over- or under-tighten on a future edit, so this harness
#   pins behavior with 26 cases.
#
# RUN LOCALLY
#   bash .claude/hooks/secret-guard.test.sh
#   bash .claude/hooks/secret-guard.test.sh /path/to/alt/secret-guard.sh
#   Exit code = number of failed cases (0 = all green).
#
# WHY THE TEST PAYLOADS LIVE IN A SCRIPT FILE (NOT INLINE BASH)
#   The trip-wire patterns (cat foo.pem, terraform.tfvars, etc.) need to
#   appear inside the test arguments. If those tokens were on the Bash
#   command line, the hook itself would deny the test invocation. By
#   keeping the patterns inside this file, a normal `bash <this-file>`
#   invocation has no trip-wires on its command line — only inside the
#   file's source, which Bash never inspects.

set -u

HOOK="${1:-$(cd "$(dirname "$0")" && pwd)/secret-guard.sh}"
if [[ ! -x "$HOOK" ]]; then
  echo "FATAL: hook script not found or not executable: $HOOK" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "FATAL: jq is required" >&2
  exit 2
fi

pass=0
fail=0

run_test() {
  local label="$1" payload="$2" expect="$3"
  local out got mark
  out="$(printf '%s' "$payload" | "$HOOK")"
  if [[ -z "$out" ]]; then got="ALLOW"; else got="DENY"; fi
  if [[ "$got" == "$expect" ]]; then mark="PASS"; pass=$((pass+1)); else mark="FAIL"; fail=$((fail+1)); fi
  printf '%-4s  expect=%-5s got=%-5s  %s\n' "$mark" "$expect" "$got" "$label"
  if [[ "$mark" == "FAIL" && -n "$out" ]]; then
    printf '       hook reason: %s\n' "$(printf '%s' "$out" | jq -r '.hookSpecificOutput.permissionDecisionReason')"
  fi
}

echo "--- Read tool: deny secret paths ---"
run_test "Read main.tf is allowed"            '{"tool_name":"Read","tool_input":{"file_path":"infra/aws-gpu/main.tf"}}'                  ALLOW
run_test "Read .pem is denied"                '{"tool_name":"Read","tool_input":{"file_path":"infra/aws-gpu/.ssh/key.pem"}}'             DENY
run_test "Read terraform.tfvars is denied"    '{"tool_name":"Read","tool_input":{"file_path":"infra/aws-gpu/terraform.tfvars"}}'         DENY
run_test "Read ~/.aws/credentials is denied"  '{"tool_name":"Read","tool_input":{"file_path":"/home/runner/.aws/credentials"}}'          DENY
run_test "Read ~/.aws/config is denied"       '{"tool_name":"Read","tool_input":{"file_path":"/home/runner/.aws/config"}}'               DENY

echo
echo "--- Bash tool: deny obvious credential leaks ---"
run_test "Bash ls is allowed"                 '{"tool_name":"Bash","tool_input":{"command":"ls -la"}}'                                    ALLOW
run_test "Bash view .pem is denied"           '{"tool_name":"Bash","tool_input":{"command":"cat infra/aws-gpu/.ssh/foo.pem"}}'            DENY
run_test "Bash view tfvars is denied"         '{"tool_name":"Bash","tool_input":{"command":"cat infra/aws-gpu/terraform.tfvars"}}'        DENY
run_test "Bash less ~/.aws/credentials denied" '{"tool_name":"Bash","tool_input":{"command":"less ~/.aws/credentials"}}'                  DENY
run_test "Bash base64 .pem is denied"         '{"tool_name":"Bash","tool_input":{"command":"base64 infra/aws-gpu/.ssh/key.pem"}}'         DENY
run_test "Bash copy .pem is denied"           '{"tool_name":"Bash","tool_input":{"command":"cp infra/aws-gpu/.ssh/key.pem /tmp/x"}}'      DENY
run_test "Bash iam key creation is denied"    '{"tool_name":"Bash","tool_input":{"command":"aws iam create-access-key --user-name foo"}}' DENY
run_test "Bash printenv AWS_SECRET denied"    '{"tool_name":"Bash","tool_input":{"command":"printenv AWS_SECRET_ACCESS_KEY"}}'            DENY
run_test "Bash env grep AWS_SECRET denied"    '{"tool_name":"Bash","tool_input":{"command":"env | grep AWS_SECRET_ACCESS_KEY"}}'          DENY
run_test "Bash echo \$AWS_SECRET denied"      '{"tool_name":"Bash","tool_input":{"command":"echo $AWS_SECRET_ACCESS_KEY"}}'               DENY
run_test "Bash tee to .pem is denied"         '{"tool_name":"Bash","tool_input":{"command":"curl https://example.com | tee /tmp/x.pem"}}' DENY

echo
echo "--- Bash tool: allow legitimate work ---"
run_test "Bash sts get-caller-identity OK"    '{"tool_name":"Bash","tool_input":{"command":"aws sts get-caller-identity --profile cutile"}}' ALLOW
run_test "Bash terraform plan is allowed"     '{"tool_name":"Bash","tool_input":{"command":"terraform -chdir=infra/aws-gpu plan -out=tfplan"}}' ALLOW
run_test "Bash test -f tfvars is allowed"     '{"tool_name":"Bash","tool_input":{"command":"test -f infra/aws-gpu/terraform.tfvars && echo exists"}}' ALLOW
run_test "Bash brew install is allowed"       '{"tool_name":"Bash","tool_input":{"command":"brew install awscli terraform"}}'             ALLOW

echo
echo "--- Bash tool: false-positive regressions (subcommand splitting) ---"
run_test "head + tfvars different chunks OK"  '{"tool_name":"Bash","tool_input":{"command":"aws --version | head -1; test -f infra/aws-gpu/terraform.tfvars && echo exists"}}' ALLOW
run_test "terraform plan piped to head OK"    '{"tool_name":"Bash","tool_input":{"command":"terraform plan | head -50"}}' ALLOW
run_test "echo string mentioning view+pem OK" '{"tool_name":"Bash","tool_input":{"command":"echo \"trying cat /tmp/foo.pem in description\""}}' ALLOW
run_test "ls then grep tfvars-name OK"        '{"tool_name":"Bash","tool_input":{"command":"ls infra/aws-gpu/ | grep tfvars"}}' ALLOW

echo
echo "--- Bash tool: sneaky ordering still denied ---"
run_test "cat tfvars after innocuous echo"    '{"tool_name":"Bash","tool_input":{"command":"echo hi; cat infra/aws-gpu/terraform.tfvars"}}' DENY
run_test "pipe view .pem to xxd"              '{"tool_name":"Bash","tool_input":{"command":"cat infra/aws-gpu/.ssh/key.pem | xxd"}}' DENY

echo
printf 'Result: %d pass, %d fail\n' "$pass" "$fail"
exit "$fail"
