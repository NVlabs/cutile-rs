# `.guardrails/` — optional Claude Code PreToolUse hook

These files are **not part of the cutile-rs build**. They exist for
contributors using [Claude Code](https://claude.com/claude-code) (or any
agent runtime that honors its `PreToolUse` hook contract) while running the
`infra/aws-gpu` flow. The hook denies the most common credential-leak
patterns at tool-call time — e.g. `cat *.pem`, `aws iam create-access-key`,
`printenv AWS_SECRET_ACCESS_KEY`.

## Files

| File | Purpose |
|------|---------|
| `secret-guard.sh` | The hook itself. Reads a JSON envelope on stdin, returns a `permissionDecision: "deny"` JSON object when a sensitive op is detected, else exits 0 silently. |
| `secret-guard.test.sh` | 26-case test harness pinning hook behavior (5 deny + 4 allow + 4 false-positive regressions + sneaky-ordering checks + …). |

## Wiring it up (Claude Code)

Add to `.claude/settings.json` (repo or user scope):

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

Other runtimes (Cursor, Codex, etc.) do not currently load this hook — the
hard rules in `../AGENTS.md` are the only enforcement and they're on the
user.

## Running the tests locally

```bash
bash infra/aws-gpu/.guardrails/secret-guard.test.sh
```

Exit code = number of failed cases. Requires `jq` and `bash`.

## CI

Changes to anything in `.guardrails/` are validated by
`.github/workflows/aws-gpu-guardrails.yml`. The path filter keeps the rest
of the repo's CI from being affected by this folder, and vice versa.

## Why the test payloads live in a script (not inline shell)

The trip-wire patterns (`cat foo.pem`, `terraform.tfvars`, etc.) need to
appear inside the test arguments. If those tokens were on the bash command
line, the hook itself would deny the test invocation. Keeping the patterns
inside this file means a normal `bash <this-file>` invocation has no
trip-wires on its command line — only inside the file's source, which Bash
never inspects.
