#!/bin/bash
# Install the froggeric fixed Qwen chat template in place of the stock one.
#
# Same four-line shape as eugr's `fix-qwen3.5-chat-template` mod, and the same
# destination filename, so the recipe's `--chat-template unsloth.jinja` keeps
# working unchanged. This is why the swap needs no `tokenizer_config.json`
# surgery (which is what the upstream README suggests) and never mutates the
# shared HF cache — the template is a file in the container's workspace, not
# part of the model snapshot.
#
# SOURCE, PINNED — do not float this to `main`. The repo has shipped 21
# versions and iterates fast; an unpinned template would silently change what a
# benchmark run is measuring.
#   https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates
#   revision  23a40b0bd4d197c31d39e3c442fd2cd6100b3971   (v21.3, 2026-07-03)
#   sha256    d203f3342d8a7f8474dd55563eece3a26e71b21c6f667c9db9c93b762b3bf997
#
# The `qwen3.6-` prefix in the template's own version string is legacy naming:
# v17 unified Qwen 3.5 and 3.6 into this single file, so it does cover the
# Qwen3.5 122B.
set -e
cp chat_template.jinja $WORKSPACE_DIR/unsloth.jinja
echo "=======> froggeric v21.3 chat template installed as unsloth.jinja"
