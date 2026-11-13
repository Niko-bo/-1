param([Parameter(Mandatory=$true)][string]$Root, [int]$Port = 8188)

# Start ComfyUI for an 8GB laptop GPU and wait until the API answers.
# ASCII only: PS 5.1 mis-decodes BOM-less UTF-8.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''; $env:ALL_PROXY=''; $env:NO_PROXY='*'
$log = Join-Path $Root 'out\comfy.log'
$py  = Join-Path $Root 'envs\comfy\Scripts\python.exe'
$comfy = Join-Path $Root 'ComfyUI'

function Log($m) {
    $line = ("{0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), $m)
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

# already up?
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/system_stats" -UseBasicParsing -TimeoutSec 3
    Log "server already running (status $($r.StatusCode))"
    exit 0
} catch { }

Log "starting ComfyUI on port $Port ..."
$args = @(
    'main.py',
    '--port', "$Port",
    '--lowvram',
    '--reserve-vram', '0.8',
    '--preview-method', 'none'
)
$p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $comfy `
     -RedirectStandardOutput (Join-Path $Root 'out\comfy.stdout.log') `
     -RedirectStandardError  (Join-Path $Root 'out\comfy.stderr.log') `
     -PassThru -WindowStyle Hidden
Log "pid $($p.Id)"

for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/system_stats" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) {
            $j = $r.Content | ConvertFrom-Json
            Log ("ready: comfyui " + $j.system.comfyui_version + " / device " + $j.devices[0].name + " / vram " + [math]::Round($j.devices[0].vram_total/1GB,2) + "GB")
            exit 0
        }
    } catch { }
}
Log "!! server did not become ready in 180s, see out\comfy.stderr.log"
exit 12
