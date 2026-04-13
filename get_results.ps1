<#
This script runs the Python concurrency examples, extracts timing results
for each approach, and writes a markdown performance report.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Python interpreter not found at $python. Create/activate the venv first."
    exit 1
}

$targets = @(
    @{ Approach = "Sequential";      Script = ".\sequential\sequential_fetcher.py" },
    @{ Approach = "Multithreading";  Script = ".\multithreading\threading_fetcher.py" },
    @{ Approach = "Multiprocessing"; Script = ".\multiprocessing\multiprocessing_fetcher.py" },
    @{ Approach = "Async I/O";       Script = ".\async_io\async_fetcher.py" }
)

$rows = foreach ($t in $targets) {
    Write-Host "Running $($t.Approach)..."
    $scriptPath = Join-Path $root $t.Script
    if (-not (Test-Path $scriptPath)) {
        Write-Warning "Script not found: $scriptPath"
        [pscustomobject]@{
            Approach = $t.Approach
            IO       = "N/A"
            CPU      = "N/A"
        }
        continue
    }

    # Do not stop the whole report generation if one script run fails.
    $out = & $python $scriptPath 2>&1 | Out-String

    $ioMatch  = [regex]::Match($out, "Part 1 \(I/O-Bound\):\s+([0-9.]+)")
    $cpuMatch = [regex]::Match($out, "Part 2 \(CPU-Bound\):\s+([0-9.]+)")

    [pscustomobject]@{
        Approach = $t.Approach
        IO       = if ($ioMatch.Success) { $ioMatch.Groups[1].Value } else { "N/A" }
        CPU      = if ($cpuMatch.Success) { $cpuMatch.Groups[1].Value } else { "N/A" }
    }
}

$tableRows = ($rows | ForEach-Object { "| $($_.Approach) | $($_.IO) | $($_.CPU) |" }) -join "`r`n"

$md = @"
# Performance Report

## Timing Results

| Approach | Part 1 I/O Time (s) | Part 2 CPU Time (s) |
|---|---:|---:|
$tableRows

## Analysis

- Explain why threading and async improved I/O-bound performance (overlapping wait time).
- Explain why multiprocessing was strongest for CPU-bound work (true parallelism, separate GIL per process).
- Discuss GIL impact, context switching overhead, and process startup/IPC overhead.

## Challenges

- Note any network failures/timeouts and how they were handled.
- Note thread safety concerns and use of locks where applicable.
- Note multiprocessing spawn overhead on Windows.
"@

$reportPath = Join-Path $root "PerformanceReport.md"
Set-Content -Path $reportPath -Value $md -Encoding UTF8
Write-Host "Wrote $reportPath"