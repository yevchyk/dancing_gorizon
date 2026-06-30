# Continuous Binance forward-shadow (no orders). Detached launcher:
#   powershell -NoProfile -ExecutionPolicy Bypass -File binance_shadow.ps1
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
cmd /c ".venv\Scripts\python.exe -m src.run_binance_live --shadow >> binance_shadow.log 2>&1"
Add-Content binance_shadow.log ("[{0:HH:mm:ss}] SHADOW RUNNER EXITED code=$LASTEXITCODE" -f (Get-Date))
