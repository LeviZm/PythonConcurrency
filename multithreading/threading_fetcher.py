"""
threading_fetcher.py

Approach: Multithreading (concurrent.futures.ThreadPoolExecutor)

Part 1 – I/O-Bound: Multiple threads issue HTTP requests concurrently.
         While one thread is blocked waiting for a network response, the OS
         scheduler runs other threads, overlapping wait times. This provides
         a significant speedup over the sequential approach for I/O-bound work.

Part 2 – CPU-Bound: Multiple threads run the word-frequency analysis
         concurrently. Python's Global Interpreter Lock (GIL) prevents more
         than one thread from executing Python bytecode at the same time, so
         true parallel CPU execution is NOT achieved. Expect performance
         similar to sequential for CPU-heavy work.

Race condition note: threads share the same stdout file handle. Without
coordination, log lines from concurrent threads can interleave. A
threading.Lock (_print_lock) is used to serialize print calls so that each
line is written atomically.
"""

import re
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# URL list (62 public websites, identical across all four implementations)
# ---------------------------------------------------------------------------
URL_LIST = [
    "https://www.python.org",
    "https://www.wikipedia.org",
    "https://www.github.com",
    "https://stackoverflow.com",
    "https://www.reddit.com",
    "https://www.bbc.com",
    "https://www.reuters.com",
    "https://www.nasa.gov",
    "https://www.imdb.com",
    "https://www.apple.com",
    "https://www.microsoft.com",
    "https://www.bloomberg.com",
    "https://www.techcrunch.com",
    "https://www.wired.com",
    "https://www.theverge.com",
    "https://arstechnica.com",
    "https://news.ycombinator.com",
    "https://dev.to",
    "https://www.freecodecamp.org",
    "https://www.w3schools.com",
    "https://developer.mozilla.org",
    "https://docs.python.org",
    "https://pypi.org",
    "https://www.djangoproject.com",
    "https://flask.palletsprojects.com",
    "https://fastapi.tiangolo.com",
    "https://numpy.org",
    "https://pandas.pydata.org",
    "https://matplotlib.org",
    "https://scikit-learn.org",
    "https://pytorch.org",
    "https://www.tensorflow.org",
    "https://www.kaggle.com",
    "https://www.coursera.org",
    "https://www.edx.org",
    "https://www.udemy.com",
    "https://www.khanacademy.org",
    "https://www.mit.edu",
    "https://www.stanford.edu",
    "https://www.harvard.edu",
    "https://www.ox.ac.uk",
    "https://www.cam.ac.uk",
    "https://www.whitehouse.gov",
    "https://www.congress.gov",
    "https://www.nih.gov",
    "https://www.cdc.gov",
    "https://www.who.int",
    "https://www.un.org",
    "https://www.nationalgeographic.com",
    "https://www.nature.com",
    "https://www.science.org",
    "https://www.economist.com",
    "https://www.nytimes.com",
    "https://www.washingtonpost.com",
    "https://www.theguardian.com",
    "https://www.cnn.com",
    "https://www.medium.com",
    "https://www.forbes.com",
    "https://www.wsj.com",
    "https://www.pcmag.com",
    "https://www.zdnet.com",
    "https://www.cnet.com",
]

# Number of worker threads. For I/O-bound work, using more threads than CPU
# cores is beneficial because most threads are idle waiting on the network.
MAX_WORKERS = 20

# Seconds to wait for an HTTP response before giving up on that URL.
REQUEST_TIMEOUT = 10

# ---------------------------------------------------------------------------
# Thread-safety: shared stdout requires a Lock to prevent interleaved output.
#
# Without this lock, two threads can call print() simultaneously and their
# output lines can be written in fragments, jumbling the log. Acquiring the
# lock before each print() ensures only one thread writes at a time.
# This is the simplest form of a race-condition mitigation.
# ---------------------------------------------------------------------------
_print_lock = threading.Lock()


def _safe_print(message: str) -> None:
    """Write a message to stdout in a thread-safe manner."""
    with _print_lock:
        print(message)


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def fetch_page(url: str) -> str:
    """
    Fetch the raw HTML content of a URL using a synchronous HTTP GET request.

    The requests library is thread-safe: each thread uses its own connection
    object, so no additional locking is needed here.

    Returns an empty string on any error so that one failing URL does not
    abort the entire benchmark run.
    """
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        return response.text
    except Exception:
        return ""


def extract_title(html: str) -> str:
    """
    Parse the <title> element from raw HTML using BeautifulSoup.

    Returns an empty string if the HTML is empty or no <title> tag exists.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    return title_tag.get_text(strip=True) if title_tag else ""


def cpu_bound_analysis(html: str) -> dict:
    """
    Perform a CPU-intensive analysis on raw HTML content.

    Steps:
      1. Strip HTML markup to plain text (BeautifulSoup).
      2. Tokenize: extract only alphabetic words, lowercased.
      3. Build a word frequency table with collections.Counter.
      4. Compute statistical measures on word lengths (statistics module).

    NOTE ON THE GIL: When this function runs inside a ThreadPoolExecutor,
    multiple threads attempt to execute these Python instructions in parallel.
    However, the GIL ensures only ONE thread executes Python bytecode at any
    instant. Threads take turns holding the GIL, yielding it roughly every
    100 bytecode instructions (sys.getswitchinterval). This means CPU-bound
    threading offers little to no speedup over sequential execution.

    Returns a dict with word_count, unique_words, top_10, mean_len,
    median_len, and stdev_len.
    """
    if not html:
        return {
            "word_count": 0,
            "unique_words": 0,
            "top_10": [],
            "mean_len": 0.0,
            "median_len": 0.0,
            "stdev_len": 0.0,
        }

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ")
    words = re.findall(r"[a-zA-Z]+", text.lower())

    if not words:
        return {
            "word_count": 0,
            "unique_words": 0,
            "top_10": [],
            "mean_len": 0.0,
            "median_len": 0.0,
            "stdev_len": 0.0,
        }

    freq = Counter(words)
    lengths = [len(w) for w in words]

    return {
        "word_count": len(words),
        "unique_words": len(freq),
        "top_10": freq.most_common(10),
        "mean_len": statistics.mean(lengths),
        "median_len": statistics.median(lengths),
        "stdev_len": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Worker functions (submitted to the thread pool)
# ---------------------------------------------------------------------------

def _fetch_and_extract(url: str) -> tuple:
    """Worker: fetch a page and extract its title. Returns (url, title, html)."""
    html = fetch_page(url)
    title = extract_title(html)
    _safe_print(f"  [IO]  {url[:60]:<60}  title: {title[:40]}")
    return url, title, html


def _analyze(args: tuple) -> tuple:
    """
    Worker: run CPU-bound analysis on one page. Returns (url, analysis_dict).

    Accepts a (url, html) tuple so it can be submitted with a single argument
    via executor.submit(), keeping the call signature consistent.
    """
    url, html = args
    analysis = cpu_bound_analysis(html)
    _safe_print(f"  [CPU] {url[:60]:<60}  words: {analysis['word_count']}")
    return url, analysis


# ---------------------------------------------------------------------------
# Part 1: I/O-Bound Task
# ---------------------------------------------------------------------------

def run_io_bound(urls: list) -> tuple:
    """
    Fetch HTML and extract titles using a ThreadPoolExecutor.

    All URL requests are submitted to the pool at once. Threads overlap their
    network wait times: when thread A is blocked waiting for a response,
    thread B can issue its own request. The effective wall-clock time drops
    roughly to (slowest_single_request) instead of sum(all_wait_times).

    Returns:
      results  – list of (url, title, html) tuples (order may vary)
      elapsed  – wall-clock time in seconds
    """
    results = []
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_and_extract, url): url for url in urls}
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.perf_counter() - start
    return results, elapsed


# ---------------------------------------------------------------------------
# Part 2: CPU-Bound Task
# ---------------------------------------------------------------------------

def run_cpu_bound(io_results: list) -> tuple:
    """
    Run word-frequency analysis using a ThreadPoolExecutor.

    Threads are launched concurrently but the GIL serializes Python bytecode
    execution. Expect this to perform similarly to — or slightly slower than —
    the sequential approach due to thread creation and context-switching
    overhead without the benefit of true parallelism.

    Returns:
      results  – list of (url, analysis_dict) tuples
      elapsed  – wall-clock time in seconds
    """
    results = []
    start = time.perf_counter()

    args = [(url, html) for url, _title, html in io_results]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_analyze, arg): arg[0] for arg in args}
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.perf_counter() - start
    return results, elapsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Approach: Multithreading (ThreadPoolExecutor)")
    print(f"          max_workers={MAX_WORKERS}")
    print("=" * 70)

    print("\n--- Part 1: I/O-Bound (fetching titles) ---")
    io_results, io_time = run_io_bound(URL_LIST)

    print("\n--- Part 2: CPU-Bound (word-frequency analysis) ---")
    cpu_results, cpu_time = run_cpu_bound(io_results)

    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print(f"  Part 1 (I/O-Bound):  {io_time:.4f} seconds")
    print(f"  Part 2 (CPU-Bound):  {cpu_time:.4f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
