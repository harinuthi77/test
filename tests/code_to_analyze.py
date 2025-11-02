"""
Complex Python code for analysis testing
This module contains intentional issues and patterns to analyze
"""

import time
from typing import List, Optional, Dict
from functools import lru_cache
import asyncio

class DataProcessor:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.processed_count = 0
        self._cache = {}

    def process_batch(self, items: List[dict]) -> List[dict]:
        """Process a batch of items with retry logic"""
        results = []

        for item in items:
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    result = self._process_single_item(item)
                    results.append(result)
                    self.processed_count += 1
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        print(f"Failed to process {item}: {e}")
                        results.append({"error": str(e), "item": item})
                    time.sleep(0.1 * retry_count)

        return results

    def _process_single_item(self, item: dict) -> dict:
        """Process a single item - may raise exceptions"""
        if not item:
            raise ValueError("Empty item")

        # Simulate processing
        processed = {
            "id": item.get("id"),
            "value": item.get("value", 0) * 2,
            "timestamp": time.time()
        }

        return processed

    @lru_cache(maxsize=128)
    def expensive_computation(self, n: int) -> int:
        """Expensive recursive computation with caching"""
        if n <= 1:
            return n
        return self.expensive_computation(n - 1) + self.expensive_computation(n - 2)


class AsyncDataFetcher:
    """Async data fetcher with connection pooling"""

    def __init__(self, base_url: str, max_connections: int = 10):
        self.base_url = base_url
        self.max_connections = max_connections
        self.active_connections = 0

    async def fetch_data(self, endpoint: str) -> Optional[dict]:
        """Fetch data from an endpoint asynchronously"""
        async with self._get_connection() as conn:
            try:
                await asyncio.sleep(0.1)  # Simulate network delay
                return {"status": "success", "data": f"Data from {endpoint}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

    async def _get_connection(self):
        """Context manager for connection handling"""
        class Connection:
            def __init__(self, fetcher):
                self.fetcher = fetcher

            async def __aenter__(self):
                while self.fetcher.active_connections >= self.fetcher.max_connections:
                    await asyncio.sleep(0.01)
                self.fetcher.active_connections += 1
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.fetcher.active_connections -= 1

        return Connection(self)

    async def fetch_multiple(self, endpoints: List[str]) -> List[dict]:
        """Fetch data from multiple endpoints concurrently"""
        tasks = [self.fetch_data(endpoint) for endpoint in endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results


def find_performance_bottlenecks(data: List[int]) -> Dict[str, any]:
    """Function with potential performance issues"""
    result = {}

    # Issue 1: Inefficient list concatenation
    all_squares = []
    for i in data:
        all_squares = all_squares + [i ** 2]  # Creates new list each time

    # Issue 2: Nested loops creating O(n²) complexity
    duplicates = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] == data[j]:
                duplicates.append(data[i])

    # Issue 3: Multiple passes over data
    max_val = max(data)
    min_val = min(data)
    avg_val = sum(data) / len(data)

    result["squares"] = all_squares
    result["duplicates"] = duplicates
    result["stats"] = {"max": max_val, "min": min_val, "avg": avg_val}

    return result


# Memory leak potential
global_cache = []

def leaky_function(data):
    """Function that might cause memory issues"""
    global global_cache
    # Data keeps accumulating
    global_cache.append(data)
    return len(global_cache)
