param(
    [Parameter(Mandatory=$true)][string]$Root,
    [int]$Port = 8188
)

# Phase 0 installer, reproduced from the working build on 2026-11-13.
# Requires Python 3.11 on PATH (the torch wheels below are cp311).
# ASCII only: Windows PowerShell 5.1 mis-decodes BOM-less UTF-8.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''; $env:ALL_PROXY=''; $env:NO_PROXY='*'

$log = Join-Path $Root 'out\install.log'
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'out') | Out-Null
function Log($m) {
    $line = ("{0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), $m)
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

# git's global proxy is often a dead local port; every call disables it explicitly.
$noProxy = @('-c','http.proxy=','-c','https.proxy=')
$py311 = & python -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>&1
Log "python on PATH: $py311 (needs 3.11)"
if ($py311 -ne '3.11') { Log "!! Python 3.11 required, got $py311"; exit 10 }

Log "=== install start ==="

# 1. ComfyUI source
$comfy = Join-Path $Root 'ComfyUI'
if (-not (Test-Path (Join-Path $comfy 'main.py'))) {
    Log "cloning ComfyUI via ghproxy ..."
    & git @noProxy clone --depth 1 https://ghproxy.net/https://github.com/comfyanonymous/ComfyUI.git $comfy 2>&1 |
        ForEach-Object { Log "  git: $_" }
}
if (-not (Test-Path (Join-Path $comfy 'main.py'))) { Log "!! ComfyUI clone failed"; exit 11 }
Log "ComfyUI source OK"

# 2. venv
$venv = Join-Path $Root 'envs\comfy'
$py = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $py)) {
    Log "creating venv ..."
    & python -m venv $venv
}
Log "python: $py"

# 3. torch: multi-threaded wheel download (official index throttles single connections)
$wheels = Join-Path $Root 'wheels'
New-Item -ItemType Directory -Force -Path $wheels | Out-Null
$torchWheel = Join-Path $wheels 'torch-2.9.1+cu126-cp311-cp311-win_amd64.whl'
$tvWheel    = Join-Path $wheels 'torchvision-0.24.1+cu126-cp311-cp311-win_amd64.whl'
$base = 'https://download.pytorch.org/whl/cu126'
if (-not (Test-Path $torchWheel)) {
    Log "downloading torch 2.9.1+cu126 (2.4GB) ..."
    & python (Join-Path $Root 'scripts\dl_parallel.py') "$base/torch-2.9.1%2Bcu126-cp311-cp311-win_amd64.whl" $torchWheel --conn 12 2>&1 |
        Select-Object -Last 3 | ForEach-Object { Log "  dl: $_" }
}
if (-not (Test-Path $tvWheel)) {
    & python (Join-Path $Root 'scripts\dl_parallel.py') "$base/torchvision-0.24.1%2Bcu126-cp311-cp311-win_amd64.whl" $tvWheel --conn 6 2>&1 |
        Select-Object -Last 2 | ForEach-Object { Log "  dl: $_" }
}
if ((Test-Path $torchWheel) -and (Test-Path $tvWheel)) {
    Log "installing torch wheels ..."
    & $py -m pip install --force-reinstall --no-deps $torchWheel $tvWheel --disable-pip-version-check 2>&1 |
        Select-String -Pattern 'Successfully|ERROR' | ForEach-Object { Log "  pip: $($_.Line)" }
} else {
    Log "wheel download incomplete -> falling back to pip index (slow)"
    & $py -m pip install torch==2.9.1 torchvision==0.24.1 --index-url $base --disable-pip-version-check 2>&1 |
        Select-String -Pattern 'Successfully|ERROR' | ForEach-Object { Log "  pip: $($_.Line)" }
}
Log ("torch: " + (& $py -c "import torch;print(torch.__version__, torch.cuda.is_available())" 2>&1))

# 4. ComfyUI requirements (tsinghua mirror + PyPI for packages the mirror lags on)
Log "installing ComfyUI requirements ..."
& $py -m pip install -r (Join-Path $comfy 'requirements.txt') `
    -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://pypi.org/simple `
    --disable-pip-version-check 2>&1 |
    Select-String -Pattern 'Successfully installed|ERROR' | ForEach-Object { Log "  req: $($_.Line)" }

& $py -m pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet --disable-pip-version-check 2>&1 | Out-Null

# 5. model paths: keep the 12GB of weights outside the repo
$yaml = @"
cgpipeline:
    base_path: $($Root -replace '\\','/')/models/
    checkpoints: checkpoints
    controlnet: controlnet
    ipadapter: ipadapter
    clip_vision: clip_vision
    upscale_models: upscale_models
    loras: loras
"@
Set-Content -Path (Join-Path $comfy 'extra_model_paths.yaml') -Value $yaml -Encoding UTF8
Log "extra_model_paths.yaml written"

Log "=== install done ==="
Log "next: python scripts\04_download_models.py --only core"
Log "then: powershell -File scripts\01_start_comfy.ps1 -Root $Root -Port $Port"
exit 0
