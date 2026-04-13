"""
async_fetcher.py

Approach: Asynchronous I/O (asyncio + aiohttp)

Part 1 – I/O-Bound: A single-threaded asyncio event loop manages many
         concurrent HTTP connections via aiohttp. Each `await` suspends the
         current coroutine and lets the event loop run others while waiting
         for a network response. No threads are created for I/O; the OS
         notifies the event loop when data arrives (epoll/IOCP). This provides
         excellent I/O throughput with very low overhead.

Part 2 – CPU-Bound: The synchronous cpu_bound_analysis function is submitted
         to a ProcessPoolExecutor via loop.run_in_executor(). This offloads
         CPU-intensive work to separate processes (bypassing the GIL) while
         keeping the event loop non-blocking. Calling cpu_bound_analysis
         directly inside a coroutine (without run_in_executor) would BLOCK
         the event loop for the duration of the computation, preventing all
         other coroutines from running.
"""

import asyncio
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import aiohttp
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

# Maximum concurrent outbound HTTP connections managed by the aiohttp connector.
MAX_CONNECTIONS = 20

# Number of worker processes used for the CPU-bound executor in Part 2.
MAX_WORKERS = 8

# Per-request timeout: allow up to 10 seconds for the full response.
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)


# ---------------------------------------------------------------------------
# Synchronous helper functions
#
# cpu_bound_analysis must be defined at the MODULE TOP LEVEL and be picklable
# so it can be sent to a ProcessPoolExecutor worker via run_in_executor.
# Nested functions, lambdas, and closures are NOT picklable.
# ---------------------------------------------------------------------------

def extract_title(html: str) -> str:
    """
    Parse the <title> element from raw HTML using BeautifulSoup.

    Returns an empty string if the HTML is empty or no <title> tag is found.
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

    This is a SYNCHRONOUS function. It is intentionally kept synchronous so
    it can be submitted to a ProcessPoolExecutor via loop.run_in_executor(),
    which returns an awaitable Future. This pattern avoids blocking the asyncio
    event loop during computation while still achieving true CPU parallelism
    across separate processes.

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
# Async helper: single-page fetch
# ---------------------------------------------------------------------------

async def fetch_page_async(session: aiohttp.ClientSession, url: str) -> tuple:
    """
    Asynchronously fetch the raw HTML of a single URL using aiohttp.

    The `async with session.get(...)` opens a non-blocking HTTP connection.
    The `await response.text()` suspends this coroutine at the `await` keyword,
    yielding control back to the event loop. While this coroutine waits for
    bytes to arrive from the network, the event loop can advance other
    coroutines (i.e., fetch other URLs), achieving concurrency without threads.

    Returns (url, html_string). Returns an empty string on any error.
    """
    try:
        async with session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
            ssl=False,  # skip SSL certificate verification to avoid cert errors
        ) as response:
            html = await response.text(errors="replace")
            return url, html
    except Exception:
        return url, ""


# ---------------------------------------------------------------------------
# Part 1: I/O-Bound Task
# ---------------------------------------------------------------------------

async def run_io_bound_async(urls: list) -> tuple:
    """
    Concurrently fetch HTML and extract titles using aiohttp + asyncio.gather.

    asyncio.gather() schedules all fetch coroutines concurrently. Each
    coroutine runs on the same thread; there is no OS-level thread creation.
    The event loop interleaves them cooperatively: when one awaits a response,
    another is free to proceed. This typically achieves near-identical
    throughput to multithreading for I/O-bound work with lower overhead.

    Returns:
      results  – list of (url, title, html) tuples
      elapsed  – wall-clock time in seconds
    """
    connector = aiohttp.TCPConnector(limit=MAX_CONNECTIONS)
    results = []
    start = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector) as session:
        # Build a list of coroutines (one per URL) and schedule them all at once.
        fetch_coroutines = [fetch_page_async(session, url) for url in urls]
        raw_results = await asyncio.gather(*fetch_coroutines, return_exceptions=False)

    for url, html in raw_results:
        title = extract_title(html)
        results.append((url, title, html))
        print(f"  [IO]  {url[:60]:<60}  title: {title[:40]}")

    elapsed = time.perf_counter() - start
    return results, elapsed


# ---------------------------------------------------------------------------
# Part 2: CPU-Bound Task
# ---------------------------------------------------------------------------

async def run_cpu_bound_async(io_results: list) -> tuple:
    """
    Run CPU-intensive analysis using a ProcessPoolExecutor via run_in_executor.

    loop.run_in_executor(executor, fn, arg) submits fn(arg) to the executor
    and returns an awaitable. The event loop does not block while the worker
    process runs; it can handle other tasks (or yield to other coroutines).
    asyncio.gather() waits for all worker futures concurrently.

    Worker processes each have their own Python interpreter and GIL, so
    true parallel CPU execution is achieved across multiple cores.

    Returns:
      results  – list of (url, analysis_dict) tuples
      elapsed  – wall-clock time in seconds
    """
    loop = asyncio.get_event_loop()
    results = []
    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit each html payload to a worker process; collect awaitables.
        analysis_coroutines = [
            loop.run_in_executor(executor, cpu_bound_analysis, html)
            for _url, _title, html in io_results
        ]
        analyses = await asyncio.gather(*analysis_coroutines)

    for (url, _title, _html), analysis in zip(io_results, analyses):
        results.append((url, analysis))
        print(f"  [CPU] {url[:60]:<60}  words: {analysis['word_count']}")

    elapsed = time.perf_counter() - start
    return results, elapsed


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------

async def async_main():
    print("=" * 70)
    print("Approach: Asynchronous I/O (asyncio + aiohttp)")
    print(f"          max_connections={MAX_CONNECTIONS}, cpu_workers={MAX_WORKERS}")
    print("=" * 70)

    print("\n--- Part 1: I/O-Bound (fetching titles) ---")
    io_results, io_time = await run_io_bound_async(URL_LIST)

    print("\n--- Part 2: CPU-Bound (word-frequency analysis) ---")
    cpu_results, cpu_time = await run_cpu_bound_async(io_results)

    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print(f"  Part 1 (I/O-Bound):  {io_time:.4f} seconds")
    print(f"  Part 2 (CPU-Bound):  {cpu_time:.4f} seconds")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # asyncio.run() creates a new event loop, runs async_main() until it
    # completes, then closes the loop and cleans up resources. This is the
    # recommended entry point for asyncio programs in Python 3.7+.
    asyncio.run(async_main())
