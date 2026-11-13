#!/bin/bash
# Commit with an explicit identity (no global git config on this machine). Usage: scripts/commit.sh "message"
cd "$(dirname "$0")/.." && git add -A . && git -c user.name="gwen" -c user.email="apprentiloutre.zygouade@gmail.com" commit -q -m "$1

Co-Authored-By: Claude <noreply@anthropic.com>" && git log --oneline | head -1
