<#  setup_env.ps1 — Create/update Conda env from environment.yml, activate it,
    and optionally store a 64-char hex “pepper” in .env

    Usage:
      . .\setup_env.ps1        # dot-source to keep activation in current shell
      # or
      powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
#>

param(
  [string]$EnvFile = "environment.yml"
)

# Strict/stop-on-error (similar to set -euo pipefail)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Fail($msg) {
  Write-Host $msg -ForegroundColor Red
  exit 1
}

function Invoke-Conda {
  param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
  & conda @Args
  if ($LASTEXITCODE -ne 0) {
    throw "Conda command failed: conda $($Args -join ' ') (exit $LASTEXITCODE)"
  }
}

if (-not (Test-Path -LiteralPath $EnvFile)) {
  Fail "environment.yml not found at: $EnvFile"
}

# 1) Ensure Conda is available
$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCmd) {
  Write-Host "    Conda not found in PATH." -ForegroundColor Yellow
  Write-Host "    Install it first: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
}

# Initialize Conda for PowerShell (equivalent to: eval "$(conda shell.bash hook)")
try {
  (& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
} catch {
  Write-Host "Could not initialize Conda. Try: conda init powershell, then restart PowerShell." -ForegroundColor Red
  exit 1
}

# Read env name from environment.yml (first 'name:' line)
$match = Select-String -Path $EnvFile -Pattern '^\s*name\s*:\s*(.+)$' | Select-Object -First 1
if (-not $match) { Fail "Could not find a 'name:' entry in $EnvFile." }
$EnvName = ($match.Matches[0].Groups[1].Value).Trim().Trim("'`"")  # strip quotes if present

# 2) Create or update the environment
$exists = $false
try {
  $envsObj = (conda env list --json | Out-String | ConvertFrom-Json)
  # Match by leaf folder; also treat 'base' as existing
  if ($EnvName -eq 'base') { $exists = $true }
  else {
    foreach ($path in $envsObj.envs) {
      if ((Split-Path -Leaf $path) -eq $EnvName) { $exists = $true; break }
    }
  }
} catch {
  # Fallback text parse if --json not available
  $exists = (conda env list | Select-String -Pattern "^\s*\*?\s*$([regex]::Escape($EnvName))\s" -Quiet)
}

if ($exists) {
  Write-Host "    Updating existing environment: $EnvName"
  Invoke-Conda env update -n $EnvName -f $EnvFile --prune
} else {
  Write-Host "    Creating environment: $EnvName"
  Invoke-Conda env create -f $EnvFile
}

Write-Host "    Environment ready."

# Activate it (will persist only if script is dot-sourced)
try {
  conda activate $EnvName
  Write-Host "    Activated: $EnvName"
} catch {
  Write-Host "    Activation failed inside the script. You can activate manually with:" -ForegroundColor Yellow
  Write-Host "        conda activate $EnvName"
}

# 3) Optional 64-character pepper
$answer = Read-Host "Do you have a custom 64-character hex pepper? [y/N]"
if ($answer -match '^[Yy]$') {
  $pepper = (Read-Host "Enter pepper (64 hex chars, will be stored in .env)").Trim()
  if ($pepper -match '^[0-9a-fA-F]{64}$') {
    "STUDENT_PEPPER=$pepper" | Set-Content -LiteralPath ".env" -Encoding UTF8
    Write-Host "    .env written with STUDENT_PEPPER."
  } else {
    Write-Host "    Input is not exactly 64 hexadecimal characters – skipping .env creation." -ForegroundColor Yellow
  }
} else {
  Write-Host "    No pepper provided – skipping .env creation."
}

Write-Host "Run this later to activate the environment:  conda activate $EnvName"
