#!/usr/bin/env bash
# setup_env.sh – create / update Conda env, activate it,
#                optionally store a 64‑char hex “pepper” in .env

set -euo pipefail # Debugging

ENV_FILE="environment.yml"
ENV_NAME=$(grep -m1 '^name:' "$ENV_FILE" | cut -d' ' -f2)

## 1. Ensure Conda is available
if ! command -v conda >/dev/null 2>&1; then
    echo "    Conda (or Mamba/Micromamba) not found in PATH."
    echo "    Install it first: https://docs.conda.io/en/latest/miniconda.html"
    return 1 2>/dev/null || exit 1
fi

# Initialise Conda for the current shell so that conda activate works
# shellcheck source=/dev/null
# Initialise Conda for this *non-interactive* shell
if conda --version >/dev/null 2>&1; then
    # Conda ≥4.4 provides the shell hook
    eval "$(conda shell.bash hook)" || {
        echo "Could not initialise Conda – install Miniconda or update Conda."
        return 1 2>/dev/null || exit 1
    }
else
    echo "Conda not found in PATH."
    return 1 2>/dev/null || exit 1
fi

## 2. Create or update the environment
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "    Updating existing environment: $ENV_NAME"
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    echo "    Creating environment: $ENV_NAME"
    conda env create --file="$ENV_FILE"
fi

echo "    Environment ready."

conda activate "$ENV_NAME"
echo "    Activated: $ENV_NAME"

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

echo "Run the following command to activate the environment: conda activate $ENV_NAME"
