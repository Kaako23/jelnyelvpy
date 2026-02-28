# Build a single-file executable for Windows using PyInstaller.
# The executable starts the Gradio server and opens the default browser.
# Usage: .\scripts\build_pyinstaller.ps1
# Output: dist\jelnyelv.exe

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
Write-Host "==> Build complete. Executable: dist\Jelnyelv.exe"
Write-Host "    Run: .\dist\Jelnyelv.exe"
Write-Host ""
