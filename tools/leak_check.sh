#!/usr/bin/env bash
# Leak check: personal paths, chat ids, e-mail addresses and tokens must never reach a tracked file.
#
#   tools/leak_check.sh          check the staged files (the pre-commit hook)
#   tools/leak_check.sh --all    check every tracked and untracked, not ignored file (CI, by hand)
#
# Install as the pre-commit hook (once per clone):
#   ln -sf ../../tools/leak_check.sh .git/hooks/pre-commit
#
# Extra patterns for one machine (your name, your chat ids, ...) go one regex per line in
# .leak-patterns.local (gitignored). gitleaks also runs when it is installed.
# The patterns below are written so that this file does not match itself.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

patterns=(
    '/[D]ata/'                                  # data disks
    '/[h]ome/[a-z]'                             # home directories
    '@[g][.]us'                                 # WhatsApp group ids
    '@[s][.]whatsapp'                           # WhatsApp user ids
    '@[c][.]us\b'
    'gmail[.]com'                               # personal e-mail
    '1203[6]3[0-9]{6,}'                         # WhatsApp group id numbers
    '\+3[7]2 ?[0-9]{7,8}'                       # phone numbers
    'github_[p]at_[A-Za-z0-9_]{20,}'            # GitHub tokens
    'gh[pousr]_[A-Za-z0-9]{36}'
    'h[f]_[A-Za-z0-9]{30,}'                     # Hugging Face tokens
    's[k]-ant-[A-Za-z0-9_-]{20,}'               # Anthropic keys
    's[k]-(proj-)?[A-Za-z0-9]{32,}'             # OpenAI-style keys
    'A[K]IA[0-9A-Z]{16}'                        # AWS keys
    '-----BEGIN [A-Z ]*PRIVATE [K]EY-----'
    'xox[b]-[0-9A-Za-z-]{20,}'                  # Slack tokens
)
if [ -f .leak-patterns.local ]; then
    while IFS= read -r line; do
        [ -n "$line" ] && [ "${line:0:1}" != "#" ] && patterns+=("$line")
    done < .leak-patterns.local
fi
regex="$(IFS='|'; echo "${patterns[*]}")"

if [ "${1:-}" = "--all" ]; then
    mapfile -t files < <(git ls-files --cached --others --exclude-standard)
    show() { cat -- "$1"; }
else
    mapfile -t files < <(git diff --cached --name-only --diff-filter=ACMR)
    show() { git show ":$1"; }
fi

found=0
for f in "${files[@]}"; do
    [ -f "$f" ] || continue
    case "$f" in local.toml|local.toml.*|.leak-patterns.local) echo "leak-check: $f must stay untracked" >&2; found=1; continue ;; esac
    if hits="$(show "$f" | sed -e 's#/home/user/#/HOMEUSER/#g' -e 's#/[h]ome/r[y]an/#/HOMEUSER/#g' | grep -nIE -- "$regex" | cut -c1-200)" && [ -n "$hits" ]; then
        while IFS= read -r h; do echo "leak-check: $f:$h" >&2; done <<<"$hits"
        found=1
    fi
done

if command -v gitleaks >/dev/null; then
    if [ "${1:-}" = "--all" ]; then  # the same files (never ignored ones like local.toml), then the history
        tmp="$(mktemp -d)"
        trap 'rm -rf "$tmp"' EXIT
        printf '%s\0' "${files[@]}" | xargs -0 -r ls -d 2>/dev/null | tar -cf - -T - | tar -xf - -C "$tmp"
        gitleaks dir --no-banner --redact --log-level error "$tmp" >&2 || found=1
        if git rev-parse -q --verify HEAD >/dev/null; then
            gitleaks git --no-banner --redact --log-level error . >&2 || found=1
        fi
    else
        gitleaks git --staged --no-banner --redact --log-level error . >&2 || found=1
    fi
    echo "leak-check: gitleaks $(gitleaks version) ran" >&2
fi

if [ "$found" -ne 0 ]; then
    echo "leak-check: blocked (${#files[@]} files checked). Move machine-specific values to local.toml." >&2
    exit 1
fi
echo "leak-check: ok (${#files[@]} files checked)"
