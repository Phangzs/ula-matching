#!/usr/bin/env bash
# setup_env.sh – create / update Micromamba env, activate it,
#                optionally store a 64‑char hex “pepper” in .env

set -euo pipefail   # Debugging

ENV_FILE="environment.yml"
ENV_NAME=$(grep -m1 '^name:' "$ENV_FILE" | cut -d' ' -f2)
SOLVE_FLAGS=(--channel-priority flexible)


# 1. Ensure Micromamba is available
if ! command -v micromamba >/dev/null 2>&1; then
    echo "    Micromamba not found – bootstrapping to ~/.local/bin..."
    ARCH=$(uname -m)
    curl -L "https://micro.mamba.pm/api/micromamba/${ARCH}/latest" \
        | tar -xj -C "$HOME/.local/bin" bin/micromamba  || {
            echo "    Download failed – please install micromamba manually."
            return 1 2>/dev/null || exit 1
        }
    export PATH="$HOME/.local/bin:$PATH"
fi

# Initialise Micromamba for the current shell so that `micromamba activate` works
eval "$(micromamba shell hook --shell bash)"

## 2. Create or update the environment
if micromamba env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "    Updating existing environment: $ENV_NAME"
    micromamba env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    echo "    Creating environment: $ENV_NAME"
    micromamba env create -f "$ENV_FILE" "${SOLVE_FLAGS[@]}"
fi

echo "    Environment ready."

## 3. Optional 64‑character pepper
read -r -p "Do you have a custom 64‑character hex pepper? [y/N] " ANSWER
ANSWER=${ANSWER:-N}

if [[ "$ANSWER" =~ ^[Yy]$ ]]; then
    read -r -p "Enter pepper (64 hex chars, will be stored in .env): " PEPPER
    if [[ "$PEPPER" =~ ^[0-9a-fA-F]{64}$ ]]; then
        echo "STUDENT_PEPPER=$PEPPER" > .env
        echo "    .env written with STUDENT_PEPPER."
    else
        echo "    Input is not exactly 64 hexadecimal characters – skipping .env creation."
    fi
else
    echo "    No pepper provided – skipping .env creation."
fi


echo "Run the following command to activate the environment: micromamba activate $ENV_NAME"
