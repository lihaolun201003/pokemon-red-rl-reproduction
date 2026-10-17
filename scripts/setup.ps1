param([ValidateSet('cuda','cpu')][string]$Device='cuda')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
function Check-Exit { if($LASTEXITCODE -ne 0){throw "Previous native command failed: $LASTEXITCODE"} }
if(!(Test-Path -LiteralPath '.venv/Scripts/python.exe')){
    py -3.10 -m venv .venv
    Check-Exit
}
$pythonExe=Join-Path (Get-Location) '.venv/Scripts/python.exe'
& $pythonExe -c "import sys; assert sys.version_info[:2] == (3,10), sys.version"
Check-Exit
$wheelIndex=if($Device -eq 'cuda'){'https://download.pytorch.org/whl/cu121'}else{'https://download.pytorch.org/whl/cpu'}
$torchBuild=if($Device -eq 'cuda'){'torch==2.5.1+cu121'}else{'torch==2.5.1+cpu'}
& $pythonExe -m pip install $torchBuild --index-url $wheelIndex
Check-Exit
& $pythonExe -m pip install -r requirements-phase1.txt
Check-Exit
& $pythonExe -m pip install -e . --no-deps --no-build-isolation
Check-Exit
& $pythonExe scripts/doctor.py
Check-Exit
