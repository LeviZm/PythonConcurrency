"""
sequential_fetcher.py

Approach: Sequential (single-threaded, synchronous)

Part 1 – I/O-Bound: Fetches HTML and extracts the <title> tag from each URL
         one at a time, blocking until each HTTP response is received before
         moving to the next URL.

Part 2 – CPU-Bound: Runs word-frequency counting and statistical analysis on
         each fetched page one at a time in a plain for-loop.

This script serves as the performance baseline. All three concurrent
implementations should be compared against the times reported here.
"""

import re
import statistics
import time
from collections import Counter

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# URL list (62 public websites used across all four implementations)
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

# Seconds to wait for an HTTP response before giving up on that URL.
REQUEST_TIMEOUT = 10


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def fetch_page(url: str) -> str:
    """
    Fetch the raw HTML content of a URL using a synchronous HTTP GET request.

    A 'User-Agent' header is included because many sites return 403 or
    redirect bot traffic when no user agent is present.

    Returns an empty string if the request fails, times out, or returns a
    non-2xx status code, so that one bad URL does not abort the whole run.
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
        # Network errors, timeouts, and HTTP error responses are silently
        # skipped to keep the benchmark running through intermittent failures.
        return ""


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
    Perform a CPU-intensive analysis on raw HTML content.

    Steps:
      1. Strip all HTML markup to get plain text (BeautifulSoup).
      2. Tokenize: extract only alphabetic words, converted to lowercase.
      3. Count word frequency with collections.Counter.
      4. Compute statistical measures on word lengths using the statistics module.

    Returns a dict containing:
      word_count   – total number of words found in the document
      unique_words – number of distinct word types
      top_10       – list of (word, count) for the 10 most frequent words
      mean_len     – arithmetic mean of word lengths
      median_len   – median word length
      stdev_len    – standard deviation of word lengths
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
# Part 1: I/O-Bound Task
# ---------------------------------------------------------------------------

def run_io_bound(urls: list) -> tuple:
    """
    Sequentially fetch HTML and extract the <title> from each URL.

    Each URL is processed one at a time: the program blocks on the network
    request and does nothing else while waiting for a response. This is the
    slowest possible approach for I/O-bound work and is used as the baseline.

    Returns:
      results  – list of (url, title, html) tuples
      elapsed  – wall-clock time in seconds for the entire loop
    """
    results = []
    start = time.perf_counter()

    for url in urls:
        html = fetch_page(url)
        title = extract_title(html)
        results.append((url, title, html))
        print(f"  [IO]  {url[:60]:<60}  title: {title[:40]}")

    elapsed = time.perf_counter() - start
    return results, elapsed


# ---------------------------------------------------------------------------
# Part 2: CPU-Bound Task
# ---------------------------------------------------------------------------

def run_cpu_bound(io_results: list) -> tuple:
    """
    Sequentially run CPU-intensive word-frequency analysis on fetched pages.

    Processes each page one at a time in a plain for-loop. Because this is
    single-threaded and single-process, it is the baseline for CPU-bound work.

    Returns:
      results  – list of (url, analysis_dict) tuples
      elapsed  – wall-clock time in seconds for the entire loop
    """
    results = []
    start = time.perf_counter()

    for url, _title, html in io_results:
        analysis = cpu_bound_analysis(html)
        results.append((url, analysis))
        print(f"  [CPU] {url[:60]:<60}  words: {analysis['word_count']}")

    elapsed = time.perf_counter() - start
    return results, elapsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Approach: Sequential")
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
