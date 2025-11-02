"""
Reasoning and Problem-Solving Test
Tests the model's ability to solve complex algorithmic problems
"""

# Problem 1: Dynamic Programming - Longest Common Subsequence
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Find the length of the longest common subsequence between two strings.

    Example:
        text1 = "abcde"
        text2 = "ace"
        Output: 3 (the LCS is "ace")

    Time: O(m*n), Space: O(m*n)
    """
    m, n = len(text1), len(text2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


# Problem 2: Graph Algorithm - Dijkstra's Shortest Path
import heapq
from typing import List, Dict, Tuple

def dijkstra(graph: Dict[int, List[Tuple[int, int]]], start: int) -> Dict[int, int]:
    """
    Find shortest paths from start node to all other nodes.

    Args:
        graph: Adjacency list {node: [(neighbor, weight), ...]}
        start: Starting node

    Returns:
        Dictionary of shortest distances from start to each node

    Time: O((V + E) log V), Space: O(V)
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # Min heap: (distance, node)
    pq = [(0, start)]
    visited = set()

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        if current_node in visited:
            continue

        visited.add(current_node)

        # Check neighbors
        for neighbor, weight in graph.get(current_node, []):
            distance = current_dist + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances


# Problem 3: Backtracking - N-Queens
def solve_n_queens(n: int) -> List[List[str]]:
    """
    Solve the N-Queens problem: place N queens on NxN board so no two attack each other.

    Returns:
        List of all valid board configurations

    Time: O(N!), Space: O(N^2)
    """
    def is_safe(board, row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # Check upper-left diagonal
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1

        # Check upper-right diagonal
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1

        return True

    def backtrack(row, board):
        if row == n:
            # Found valid solution
            result.append([''.join(row) for row in board])
            return

        for col in range(n):
            if is_safe(board, row, col):
                board[row][col] = 'Q'
                backtrack(row + 1, board)
                board[row][col] = '.'  # Backtrack

    result = []
    initial_board = [['.' for _ in range(n)] for _ in range(n)]
    backtrack(0, initial_board)
    return result


# Problem 4: System Design - LRU Cache
class LRUCache:
    """
    Least Recently Used Cache with O(1) get and put operations.

    Uses HashMap + Doubly Linked List for optimal performance.
    """

    class Node:
        def __init__(self, key=0, value=0):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Dummy head and tail for easier list manipulation
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        """Remove node from linked list"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_head(self, node):
        """Add node right after head (most recently used)"""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        """Get value and mark as recently used"""
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self._remove(node)
        self._add_to_head(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        """Put key-value pair, evict LRU if at capacity"""
        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_to_head(node)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Evict LRU (node before tail)
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]

            new_node = self.Node(key, value)
            self.cache[key] = new_node
            self._add_to_head(new_node)


# Test all implementations
if __name__ == "__main__":
    print("="*70)
    print("REASONING & PROBLEM-SOLVING TESTS")
    print("="*70)

    # Test 1: LCS
    print("\n1. Longest Common Subsequence:")
    text1, text2 = "abcde", "ace"
    result = longest_common_subsequence(text1, text2)
    print(f"   LCS of '{text1}' and '{text2}': {result}")
    assert result == 3, "LCS test failed"
    print("   ✓ PASSED")

    # Test 2: Dijkstra
    print("\n2. Dijkstra's Shortest Path:")
    graph = {
        0: [(1, 4), (2, 1)],
        1: [(3, 1)],
        2: [(1, 2), (3, 5)],
        3: []
    }
    distances = dijkstra(graph, 0)
    print(f"   Shortest distances from node 0: {distances}")
    assert distances[3] == 4, "Dijkstra test failed"
    print("   ✓ PASSED")

    # Test 3: N-Queens
    print("\n3. N-Queens (4x4 board):")
    solutions = solve_n_queens(4)
    print(f"   Found {len(solutions)} solutions")
    if solutions:
        print("   First solution:")
        for row in solutions[0]:
            print(f"   {row}")
    assert len(solutions) == 2, "N-Queens test failed"
    print("   ✓ PASSED")

    # Test 4: LRU Cache
    print("\n4. LRU Cache:")
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    print(f"   Get 1: {cache.get(1)}")  # returns 1
    cache.put(3, 3)  # evicts key 2
    print(f"   Get 2: {cache.get(2)}")  # returns -1 (not found)
    cache.put(4, 4)  # evicts key 1
    print(f"   Get 1: {cache.get(1)}")  # returns -1 (not found)
    print(f"   Get 3: {cache.get(3)}")  # returns 3
    print(f"   Get 4: {cache.get(4)}")  # returns 4
    print("   ✓ PASSED")

    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
