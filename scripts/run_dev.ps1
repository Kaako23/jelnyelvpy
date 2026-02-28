# Run the jelnyelv app in dev mode.
# Activate venv if present, then start the Gradio app.
# Usage: .\scripts\run_dev.ps1

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }

Set-Location $ProjectRoot

$VenvActivate = Join-Path $ProjectRoot $VenvDir "Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
}

& python -m jelnyelv.main @args
