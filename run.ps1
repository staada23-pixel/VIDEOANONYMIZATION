# ============================================================
#  Video Anonymizer — one-shot runner (PowerShell)
# ============================================================
#  Pouziti:
#    .\run.ps1
#    .\run.ps1 --input video.mp4
#    .\run.ps1 --input video.mp4 --detector face --anon-method mosaic
#    .\run.ps1 --input "C:\cesta s mezerou\video.mp4" --no-display
# ============================================================

$ErrorActionPreference = "Stop"

# UTF-8 pro konzoli i Python (dulezite pro cesky vystup)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$ProjectRoot = $PSScriptRoot
if ($ProjectRoot.EndsWith("\")) { $ProjectRoot = $ProjectRoot.TrimEnd("\") }

# Najdi LPM Python wrappery (er.py + lpm.py)
$lpmWrappers = $null

# 1) sibling ../LPM-*/wrappers/python
$lpmDirs = Get-ChildItem -Path (Join-Path $ProjectRoot "..") -Filter "LPM-*" -Directory -ErrorAction SilentlyContinue
foreach ($d in $lpmDirs) {
    $candidate = Join-Path $d.FullName "wrappers\python"
    if (Test-Path (Join-Path $candidate "lpm.py")) {
        $lpmWrappers = $candidate
        break
    }
}

# 2) externals/LPM/wrappers/python
if (-not $lpmWrappers) {
    $candidate = Join-Path $ProjectRoot "externals\LPM\wrappers\python"
    if (Test-Path (Join-Path $candidate "lpm.py")) {
        $lpmWrappers = $candidate
    }
}

# 3) hardcoded fallback (pro tohoto uzivatele)
if (-not $lpmWrappers) {
    $candidate = "C:\Users\face\Desktop\Praxe 2026\EYEDEA PROJECT\LPM-v7.9.1-2026-04-08-Windows-10-x64-hasp10.2\wrappers\python"
    if (Test-Path (Join-Path $candidate "lpm.py")) {
        $lpmWrappers = $candidate
    }
}

# Nastav PYTHONPATH (i bez LPM to funguje - fallback detektory)
if ($lpmWrappers) {
    Write-Host "[OK] LPM wrappers: $lpmWrappers" -ForegroundColor Green
    $env:PYTHONPATH = "$ProjectRoot\src;$lpmWrappers"
} else {
    Write-Host "[INFO] LPM wrappery nedostupne, fallback na face DNN" -ForegroundColor Yellow
    $env:PYTHONPATH = "$ProjectRoot\src"
}

# Spust z project root (kvuli relativnim cestam k souborum)
Set-Location $ProjectRoot

# Spust Python - argumenty s mezerami musi byt v uvozovkach,
# PowerShell tohle resi sam, ale explicitne pro jistotu
$argList = @()
foreach ($a in $args) {
    $argList += $a
}

& python -m video_anonymizer @argList
exit $LASTEXITCODE
