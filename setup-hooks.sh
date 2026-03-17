#!/bin/sh
# Run this once after cloning the repo to activate git hooks.
# sh setup-hooks.sh

git config core.hooksPath .githooks
chmod +x .githooks/pre-push
echo "Git hooks activated. Tests will run before every push."
