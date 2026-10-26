#!/usr/bin/env bash
# Post-commit hook: a commit that touches results/*.json refreshes the live results board on the blog
# (tools/live_board.py --publish). Runs in the background; a failure only lands in state/live_board.log
# and never affects the commit. Does nothing without [board] blog_repo in local.toml.
#
# Install as the post-commit hook (once per clone):
#   ln -sf ../../tools/post_commit_board.sh .git/hooks/post-commit
cd "$(git rev-parse --show-toplevel)" || exit 0
git diff-tree --no-commit-id --name-only -r HEAD | grep -q '^results/.*\.json$' || exit 0
mkdir -p state
( python3 tools/live_board.py --publish >> state/live_board.log 2>&1 ) >/dev/null 2>&1 &
exit 0
