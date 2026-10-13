# kor_allt.ps1 - runs the main SEB analysis (three scripts) and writes one log.
#
#     powershell -ExecutionPolicy Bypass -File .\kor_allt.ps1          # everything, about 20 minutes
#     powershell -ExecutionPolicy Bypass -File .\kor_allt.ps1 -Quick   # skips lcurve + forward/backward test, about 5 minutes

param([switch]$Quick)

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:MPLBACKEND = "Agg"

Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot ("kor_allt_" + (Get-Date -Format "yyyy-MM-dd_HHmm") + ".log")
function Say([string]$t) { Write-Host $t; Add-Content -Path $log -Value $t -Encoding UTF8 }

# A working Python (the Microsoft Store 'python' stub does not count)
$py = $null
foreach ($cand in @("python", "py")) {
    if (Get-Command $cand -ErrorAction SilentlyContinue) {
        & $cand -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $py = $cand; break }
    }
}
if (-not $py) { Say "ERROR: no working Python found. Install Python 3 and tick 'Add python.exe to PATH'."; exit 1 }

$main = @("seb_merton_analys.py")
if (-not $Quick) { $main += "--full" }
$steps = @(
    @{ Name = "Main analysis (BS vs Merton)";    Args = $main },
    @{ Name = "Long/short gamma (hedged P&L)";   Args = @("seb_gamma_analys.py") },
    @{ Name = "Reports, gap test, spreads";      Args = @("seb_handelsrisk.py") }
)
$summary = @()
foreach ($s in $steps) {
    Say ""; Say "############ $($s.Name) ############"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $py @($s.Args) 2>&1 | ForEach-Object { Say "$_" }
    $status = if ($LASTEXITCODE -eq 0) { "OK" } else { "FAILED (exit $LASTEXITCODE)" }
    $summary += "{0,-35} {1,-18} {2,6:N0} s" -f $s.Name, $status, $sw.Elapsed.TotalSeconds
}
Say ""; Say "############ SUMMARY ############"
$summary | ForEach-Object { Say $_ }
Say "Results: $(Join-Path $PSScriptRoot 'RESULTAT')   Log: $log"
