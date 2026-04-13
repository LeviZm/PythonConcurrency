"""
multiprocessing_fetcher.py

Approach: Multiprocessing (concurrent.futures.ProcessPoolExecutor)

Part 1 – I/O-Bound: Worker processes perform HTTP fetches in parallel. Each
         process is a fully independent OS process with its own memory space,
         so there is no GIL contention. However, process creation is expensive
         (spawning, importing modules, setting up memory) compared to thread
         creation. For tasks whose bottleneck is network wait time rather than
         CPU work, this overhead typically makes multiprocessing slower than
         threading or asyncio for the I/O-bound portion.

Part 2 – CPU-Bound: Worker processes run the word-frequency analysis on
         separate CPU cores simultaneously. Because each process has its OWN
         Python interpreter and its OWN GIL, multiple processes can execute
         Python bytecode truly in parallel. On a multi-core machine this
         provides genuine speedup proportional to the number of available cores,
         making multiprocessing the best approach for CPU-intensive work.

Windows note: On Windows the default multiprocessing start method is "spawn".
The spawned child process re-imports this module from scratch. The
`if __name__ == "__main__":` guard at the bottom is therefore REQUIRED —
without it every worker would try to spawn more workers, causing a cascade
of RuntimeError exceptions.
"""

import re
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Number of worker processes. Setting this to the number of logical CPU cores
# maximises CPU parallelism for Part 2. Use os.cpu_count() at runtime if you
# want this set automatically.
MAX_WORKERS = 8

# Seconds to wait for an HTTP response before giving up on that URL.
REQUEST_TIMEOUT = 10


# ---------------------------------------------------------------------------
# Shared helper functions
#
# IMPORTANT: All functions submitted to ProcessPoolExecutor must be defined
# at the MODULE TOP LEVEL (not inside another function or class). The "spawn"
# start method on Windows pickles the function by its module-qualified name
# and re-imports it in the worker. Nested or lambda functions are not
# picklable and will raise AttributeError or PicklingError.
# ---------------------------------------------------------------------------

def fetch_page(url: str) -> str:
    """
    Fetch the raw HTML content of a URL using a synchronous HTTP GET request.

    This function runs inside a worker process. Each worker has its own
    address space and network stack, so no locking is needed.

    Returns an empty string on any error.
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
    CPU-intensive word-frequency and statistical analysis on raw HTML.

    Steps:
      1. Strip HTML markup to plain text (BeautifulSoup).
      2. Tokenize: extract only alphabetic words, lowercased.
      3. Count word frequency with collections.Counter.
      4. Compute mean, median, and stdev of word lengths (statistics module).

    PARALLELISM: When submitted to a ProcessPoolExecutor, each invocation
    runs in a separate OS process with its own Python interpreter and its own
    GIL. Multiple cores can execute this function truly simultaneously,
    providing real speedup proportional to the number of worker processes and
    available CPU cores.

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
# Worker functions (top-level, picklable — required for IPC on Windows)
# ---------------------------------------------------------------------------

def _fetch_and_extract(url: str) -> tuple:
    """
    Worker: fetch a page and extract its title.

    Runs inside a worker process. print() output is forwarded to the parent
    process's stdout by the OS pipe that ProcessPoolExecutor sets up.

    Returns (url, title, html).
    """
    html = fetch_page(url)
    title = extract_title(html)
    print(f"  [IO]  {url[:60]:<60}  title: {title[:40]}")
    return url, title, html


def _analyze(args: tuple) -> tuple:
    """
    Worker: run CPU-bound analysis on one page.

    Accepts a (url, html) tuple so a single argument can be passed via
    executor.submit(). Returns (url, analysis_dict).
    """
    url, html = args
    analysis = cpu_bound_analysis(html)
    print(f"  [CPU] {url[:60]:<60}  words: {analysis['word_count']}")
    return url, analysis


# ---------------------------------------------------------------------------
# Part 1: I/O-Bound Task
# ---------------------------------------------------------------------------

def run_io_bound(urls: list) -> tuple:
    """
    Fetch HTML and extract titles using a ProcessPoolExecutor.

    Each URL is dispatched to a worker process. Process creation overhead
    (spawning a new interpreter, importing modules, etc.) is significant and
    is incurred once per worker. For the I/O-bound task this overhead often
    makes multiprocessing slower than threading or asyncio; the result is
    included here for fair comparison across all four approaches.

    Returns:
      results  – list of (url, title, html) tuples (order may vary)
      elapsed  – wall-clock time in seconds
    """
    results = []
    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
    Run word-frequency analysis across multiple worker processes.

    Each worker process runs on a dedicated CPU core with its own GIL,
    enabling true parallel execution of Python bytecode. On a machine with N
    cores and N workers, the wall-clock time for CPU-bound work approaches
    (sequential_time / N), minus IPC serialization overhead.

    Arguments are serialized (pickled) to send to the worker and results are
    deserialized (unpickled) on return; large HTML payloads increase this cost.

    Returns:
      results  – list of (url, analysis_dict) tuples
      elapsed  – wall-clock time in seconds
    """
    results = []
    start = time.perf_counter()

    args = [(url, html) for url, _title, html in io_results]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
    print("Approach: Multiprocessing (ProcessPoolExecutor)")
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


# REQUIRED on Windows: the "spawn" start method re-imports this module inside
# each worker process. Without this guard the worker would call main() again,
# attempt to create another ProcessPoolExecutor, and spawn more workers
# recursively, raising RuntimeError: "An attempt has been made to start a
# new process before the current process has finished its bootstrapping phase."
if __name__ == "__main__":
    main()
