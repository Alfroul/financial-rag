"""Load test for Financial RAG FastAPI endpoints.

Usage:
    # Start API first
    python -m uvicorn src.api.app:app --port 8000

    # Run load test
    python scripts/load_test.py --base-url http://localhost:8000 --queries 10 --concurrency 5

    # With streaming
    python scripts/load_test.py --base-url http://localhost:8000 --stream --concurrency 10

Requirements:
    pip install httpx
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import httpx

# Representative financial queries
_QUERIES = [
    "什么是GDP？",
    "市盈率（PE）是什么意思？",
    "2024年A股市场IPO融资总额约为多少？",
    "沪深300指数和中证500指数有什么区别？",
    "什么是资产负债率？",
    "ROE如何计算？",
    "融资和融券有什么区别？",
    "主动型基金和被动型基金有什么区别？",
    "什么是ETF？",
    "市盈率和市净率两种估值方法有什么区别？",
]


@dataclass
class RequestResult:
    status: int
    latency_ms: float
    error: str = ""
    answer_len: int = 0
    num_sources: int = 0


@dataclass
class LoadTestReport:
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def qps(self) -> float:
        return self.successful / self.duration_s if self.duration_s > 0 else 0

    def latency_stats(self) -> dict:
        if not self.latencies:
            return {}
        s = sorted(self.latencies)
        n = len(s)
        return {
            "min_ms": round(s[0], 1),
            "p50_ms": round(s[n // 2], 1),
            "p90_ms": round(s[int(n * 0.9)], 1),
            "p95_ms": round(s[int(n * 0.95)], 1),
            "p99_ms": round(s[min(int(n * 0.99), n - 1)], 1),
            "max_ms": round(s[-1], 1),
            "avg_ms": round(statistics.mean(s), 1),
            "stdev_ms": round(statistics.stdev(s), 1) if n > 1 else 0,
        }

    def print_report(self) -> None:
        stats = self.latency_stats()
        print("\n" + "=" * 60)
        print("  LOAD TEST REPORT")
        print("=" * 60)
        print(f"  Duration:       {self.duration_s:.1f}s")
        print(f"  Total requests: {self.total_requests}")
        print(f"  Successful:     {self.successful}")
        print(f"  Failed:         {self.failed}")
        print(f"  Error rate:     {self.failed / max(self.total_requests, 1) * 100:.1f}%")
        print(f"  QPS:            {self.qps:.2f}")
        print("-" * 60)
        print("  Latency:")
        for k, v in stats.items():
            print(f"    {k:12s}  {v}")
        print("=" * 60)

        if self.errors:
            print("\n  Sample errors:")
            for e in self.errors[:5]:
                print(f"    - {e}")


def send_query(
    client: httpx.Client,
    base_url: str,
    query: str,
    use_stream: bool = False,
    timeout: float = 60.0,
) -> RequestResult:
    start = time.perf_counter()
    try:
        if use_stream:
            with client.stream(
                "POST",
                f"{base_url}/api/v1/query/stream",
                json={"question": query},
                timeout=timeout,
            ) as resp:
                chunks = list(resp.iter_text())
                status = resp.status_code
                answer = "".join(chunks)
        else:
            resp = client.post(
                f"{base_url}/api/v1/query",
                json={"question": query},
                timeout=timeout,
            )
            status = resp.status_code
            data = resp.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])

        latency = (time.perf_counter() - start) * 1000

        if status >= 400:
            return RequestResult(
                status=status, latency_ms=latency,
                error=f"HTTP {status}",
            )

        return RequestResult(
            status=status,
            latency_ms=latency,
            answer_len=len(answer),
            num_sources=len(sources) if not use_stream else 0,
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return RequestResult(status=0, latency_ms=latency, error=str(e))


def run_load_test(
    base_url: str,
    num_queries: int,
    concurrency: int,
    use_stream: bool = False,
    timeout: float = 60.0,
) -> LoadTestReport:
    report = LoadTestReport()
    report.start_time = time.perf_counter()

    # Generate query list
    queries = []
    for i in range(num_queries):
        queries.append(_QUERIES[i % len(_QUERIES)])

    print(f"Sending {num_queries} queries with concurrency={concurrency}...")
    if use_stream:
        print("  Mode: streaming (SSE)")
    else:
        print("  Mode: sync REST")

    with httpx.Client() as client, ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    send_query, client, base_url, q, use_stream, timeout,
                ): i
                for i, q in enumerate(queries)
            }

            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                report.total_requests += 1

                if result.error:
                    report.failed += 1
                    report.errors.append(f"[Q{idx}] {result.error}")
                else:
                    report.successful += 1

                report.latencies.append(result.latency_ms)

                # Progress
                done = report.total_requests
                if done % max(1, num_queries // 10) == 0 or done == num_queries:
                    print(f"  {done}/{num_queries} done "
                          f"(errors: {report.failed}, "
                          f"avg: {statistics.mean(report.latencies):.0f}ms)")

    report.end_time = time.perf_counter()
    return report


def compare_concurrency_levels(
    base_url: str,
    num_queries: int,
    levels: list[int],
    use_stream: bool = False,
) -> None:
    """Run load test at multiple concurrency levels and print comparison."""
    reports = {}
    for c in levels:
        print(f"\n--- Concurrency: {c} ---")
        reports[c] = run_load_test(
            base_url, num_queries, c, use_stream,
        )
        reports[c].print_report()

    # Print comparison table
    print("\n" + "=" * 80)
    print("  CONCURRENCY COMPARISON")
    print("=" * 80)
    print(f"  {'Concurrency':>12s} | {'QPS':>8s} | {'P50(ms)':>8s} | "
          f"{'P95(ms)':>8s} | {'P99(ms)':>8s} | {'Errors':>8s}")
    print("  " + "-" * 70)
    for c in levels:
        r = reports[c]
        s = r.latency_stats()
        print(f"  {c:>12d} | {r.qps:>8.2f} | {s.get('p50_ms', 0):>8.1f} | "
              f"{s.get('p95_ms', 0):>8.1f} | {s.get('p99_ms', 0):>8.1f} | "
              f"{r.failed:>8d}")


def main():
    parser = argparse.ArgumentParser(description="Load test for Financial RAG API")
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--queries", type=int, default=20,
        help="Total number of queries to send (default: 20)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Number of concurrent workers (default: 5)",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Use streaming SSE endpoint instead of sync",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run at multiple concurrency levels (1, 5, 10, 20, 50)",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--json-output", type=str, default="",
        help="Save report as JSON to this file",
    )
    args = parser.parse_args()

    # Health check
    try:
        resp = httpx.get(f"{args.base_url}/api/v1/health", timeout=5.0)
        if resp.status_code != 200:
            print(f"Health check failed: HTTP {resp.status_code}")
            sys.exit(1)
        print(f"API health check passed: {args.base_url}")
    except Exception as e:
        print(f"Cannot connect to API at {args.base_url}: {e}")
        print("Start the API first: python -m uvicorn src.api.app:app --port 8000")
        sys.exit(1)

    if args.compare:
        levels = [1, 5, 10, 20, 50]
        per_level = max(args.queries // len(levels), 5)
        compare_concurrency_levels(args.base_url, per_level, levels, args.stream)
    else:
        report = run_load_test(
            args.base_url, args.queries, args.concurrency, args.stream, args.timeout,
        )
        report.print_report()

        if args.json_output:
            data = {
                "config": {
                    "base_url": args.base_url,
                    "num_queries": args.queries,
                    "concurrency": args.concurrency,
                    "stream": args.stream,
                },
                "summary": {
                    "total": report.total_requests,
                    "successful": report.successful,
                    "failed": report.failed,
                    "duration_s": round(report.duration_s, 2),
                    "qps": round(report.qps, 2),
                },
                "latency": report.latency_stats(),
            }
            with open(args.json_output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\nReport saved to {args.json_output}")


if __name__ == "__main__":
    main()
