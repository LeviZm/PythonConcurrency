#!/usr/bin/env bash
set -euo pipefail

# Run all four concurrency implementations, extract timing lines,
# and generate PerformanceReport.md in the project root.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "Error: python3 not found and .venv/bin/python does not exist." >&2
  exit 1
fi

declare -a TARGETS=(
  "Sequential|./sequential/sequential_fetcher.py"
  "Multithreading|./multithreading/threading_fetcher.py"
  "Multiprocessing|./multiprocessing/multiprocessing_fetcher.py"
  "Async I/O|./async_io/async_fetcher.py"
)

rows=()

for target in "${TARGETS[@]}"; do
  approach="${target%%|*}"
  script="${target#*|}"

  echo "Running ${approach}..."

  # Capture stdout and stderr for regex extraction.
  output="$($PYTHON "$script" 2>&1 || true)"

  io_time="$(printf "%s\n" "$output" | sed -nE 's/.*Part 1 \(I\/O-Bound\):[[:space:]]+([0-9.]+).*/\1/p' | tail -n 1)"
  cpu_time="$(printf "%s\n" "$output" | sed -nE 's/.*Part 2 \(CPU-Bound\):[[:space:]]+([0-9.]+).*/\1/p' | tail -n 1)"

  [[ -n "$io_time" ]] || io_time="N/A"
  [[ -n "$cpu_time" ]] || cpu_time="N/A"

  rows+=("| ${approach} | ${io_time} | ${cpu_time} |")
done

{
  echo "# Performance Report"
  echo
  echo "## Timing Results"
  echo
  echo "| Approach | Part 1 I/O Time (s) | Part 2 CPU Time (s) |"
  echo "|---|---:|---:|"
  for row in "${rows[@]}"; do
    echo "$row"
  done
  echo
  echo "## Analysis"
  echo
  echo "- Explain why threading and async improved I/O-bound performance (overlapping wait time)."
  echo "- Explain why multiprocessing was strongest for CPU-bound work (true parallelism, separate GIL per process)."
  echo "- Discuss GIL impact, context switching overhead, and process startup/IPC overhead."
  echo
  echo "## Challenges"
  echo
  echo "- Note any network failures/timeouts and how they were handled."
  echo "- Note thread safety concerns and use of locks where applicable."
  echo "- Note multiprocessing spawn overhead on Windows/Linux process model differences."
} > PerformanceReport.md

echo "Wrote ./PerformanceReport.md"
