# Build a folder distribution with PyInstaller (onedir; avoids one-file size limits).
# The executable starts the Gradio server and opens the default browser.
# Usage: .\scripts\build_pyinstaller.ps1
# Output: dist\Jelnyelv\Jelnyelv.exe

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }

Set-Location $ProjectRoot

$VenvActivate = Join-Path $ProjectRoot $VenvDir "Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
}

pip install -e ".[dev]" 2>$null

Write-Host "==> Building Jelnyelv executable"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, dist

pyinstaller -y --clean Jelnyelv.spec

Write-Host ""
Write-Host "==> Build complete. Run: .\dist\Jelnyelv\Jelnyelv.exe"
Write-Host ""
