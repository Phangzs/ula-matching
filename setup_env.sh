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

# Initialise Conda for the current shell so that `conda activate` works
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

## 2. Create or update the environment
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "    Updating existing environment: $ENV_NAME"
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    echo "    Creating environment: $ENV_NAME"
    conda env create -f "$ENV_FILE"
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





























#!/usr/bin/env bash                            # L1
# setup_env.sh – create / update Conda env,    # L2
#                optionally store a pepper      # L3

set -euo pipefail                              # L4

ENV_FILE="environment.yml"                     # L5
ENV_NAME=$(grep -m1 '^name:' "$ENV_FILE" | cut -d' ' -f2)   # L6

## 1. Ensure Conda is available ------------------------------------ # L7
if ! command -v conda >/dev/null 2>&1; then     # L8
    echo "❌  Conda (or Mamba/Micromamba) not found in PATH." # L9
    echo "    Install it first: https://docs.conda.io/en/latest/miniconda.html" # L10
    return 1 2>/dev/null || exit 1             # L11
fi                                             # L12

# Initialise Conda for the current shell --------------------------- # L13
# shellcheck source=/dev/null                                       # L14
source "$(conda info --base)/etc/profile.d/conda.sh"                # L15

## 2. Create or update the environment ----------------------------- # L16
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then  # L17
    echo "🔄  Updating existing environment: $ENV_NAME"             # L18
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune          # L19
else                                                                # L20
    echo "➕  Creating environment: $ENV_NAME"                       # L21
    conda env create -f "$ENV_FILE"                                 # L22
fi                                                                  # L23

echo "✅  Environment ready."                                        # L24

conda activate "$ENV_NAME"                                          # L25
echo "🔹  Activated: $ENV_NAME"                                      # L26

## 3. Optional 64‑character pepper --------------------------------- # L27
read -r -p "Do you have a custom 64‑character hex pepper? [y/N] " ANSWER  # L28
ANSWER=${ANSWER:-N}                                                 # L29

if [[ "$ANSWER" =~ ^[Yy]$ ]]; then                                 # L30
    read -r -p "Enter pepper (64 hex chars, will be stored in .env): " PEPPER  # L31
    if [[ "$PEPPER" =~ ^[0-9a-fA-F]{64}$ ]]; then                  # L32
        echo "STUDENT_PEPPER=$PEPPER" > .env                       # L33
        echo "📝  .env written with STUDENT_PEPPER."                # L34
    else                                                           # L35
        echo "❗  Input is not exactly 64 hexadecimal characters – skipping .env creation."  # L36
    fi                                                             # L37
else                                                               # L38
    echo "ℹ️  No pepper provided – skipping .env creation."         # L39
fi                                                                 # L40
