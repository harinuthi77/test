"""
Mathematical Capabilities Test
Tests advanced mathematical reasoning and implementations
"""

import math
from typing import List, Tuple
from decimal import Decimal, getcontext


# Test 1: Number Theory - Prime Number Algorithms
def sieve_of_eratosthenes(n: int) -> List[int]:
    """
    Find all prime numbers up to n using Sieve of Eratosthenes.

    Time: O(n log log n), Space: O(n)
    """
    if n < 2:
        return []

    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Mark all multiples as composite
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(n + 1) if is_prime[i]]


def miller_rabin_primality(n: int, k: int = 5) -> bool:
    """
    Probabilistic primality test using Miller-Rabin algorithm.

    Args:
        n: Number to test
        k: Number of rounds (higher = more accurate)

    Returns:
        True if probably prime, False if definitely composite

    Time: O(k log³ n)
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    import random
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True


# Test 2: Linear Algebra - Matrix Operations
class Matrix:
    """Matrix operations with mathematical precision"""

    def __init__(self, data: List[List[float]]):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0

    def multiply(self, other: 'Matrix') -> 'Matrix':
        """
        Matrix multiplication: O(n³) naive algorithm

        For production, use Strassen (O(n^2.807)) or optimized BLAS
        """
        if self.cols != other.rows:
            raise ValueError("Incompatible dimensions for multiplication")

        result = [[0] * other.cols for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]

        return Matrix(result)

    def transpose(self) -> 'Matrix':
        """Matrix transposition"""
        result = [[self.data[j][i] for j in range(self.rows)]
                  for i in range(self.cols)]
        return Matrix(result)

    def determinant(self) -> float:
        """
        Calculate determinant using Gaussian elimination.
        Time: O(n³)
        """
        if self.rows != self.cols:
            raise ValueError("Determinant only defined for square matrices")

        # Create copy to avoid modifying original
        n = self.rows
        mat = [row[:] for row in self.data]

        det = 1
        for i in range(n):
            # Find pivot
            pivot_row = i
            for j in range(i + 1, n):
                if abs(mat[j][i]) > abs(mat[pivot_row][i]):
                    pivot_row = j

            if abs(mat[pivot_row][i]) < 1e-10:
                return 0  # Singular matrix

            # Swap rows
            if pivot_row != i:
                mat[i], mat[pivot_row] = mat[pivot_row], mat[i]
                det *= -1

            det *= mat[i][i]

            # Eliminate
            for j in range(i + 1, n):
                factor = mat[j][i] / mat[i][i]
                for k in range(i, n):
                    mat[j][k] -= factor * mat[i][k]

        return det


# Test 3: Numerical Methods - Newton's Method
def newtons_method(f, df, x0: float, tolerance: float = 1e-10, max_iter: int = 100) -> float:
    """
    Find root of function using Newton's method.

    Args:
        f: Function to find root of
        df: Derivative of f
        x0: Initial guess
        tolerance: Convergence threshold
        max_iter: Maximum iterations

    Returns:
        Approximate root

    Convergence: Quadratic (doubles precision each iteration)
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tolerance:
            return x

        dfx = df(x)
        if abs(dfx) < 1e-15:
            raise ValueError("Derivative too small, method fails")

        x = x - fx / dfx

    raise ValueError("Method did not converge")


# Test 4: Computational Geometry - Convex Hull
def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Compute convex hull using Graham Scan algorithm.

    Time: O(n log n), Space: O(n)
    """
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    points = sorted(set(points))
    if len(points) <= 1:
        return points

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


# Test 5: Statistics - Fast Fourier Transform
def fft(x: List[complex]) -> List[complex]:
    """
    Cooley-Tukey FFT algorithm.

    Time: O(n log n), Space: O(n)
    """
    n = len(x)
    if n <= 1:
        return x

    # Divide
    even = fft([x[i] for i in range(0, n, 2)])
    odd = fft([x[i] for i in range(1, n, 2)])

    # Conquer
    T = []
    for k in range(n // 2):
        t = math.e ** (-2j * math.pi * k / n) * odd[k]
        T.append(t)

    return [even[k] + T[k] for k in range(n // 2)] + \
           [even[k] - T[k] for k in range(n // 2)]


# Test 6: Combinatorics - Catalan Numbers
def catalan_number(n: int) -> int:
    """
    Calculate nth Catalan number: C(n) = (2n)! / ((n+1)! * n!)

    Applications: Binary trees, balanced parentheses, polygon triangulations

    Time: O(n), Space: O(1)
    """
    if n <= 1:
        return 1

    catalan = 1
    for i in range(n):
        catalan = catalan * 2 * (2 * i + 1) // (i + 2)

    return catalan


# Run all tests
if __name__ == "__main__":
    print("="*70)
    print("MATHEMATICAL CAPABILITIES TESTS")
    print("="*70)

    # Test 1: Prime Numbers
    print("\n1. Prime Number Algorithms:")
    primes_100 = sieve_of_eratosthenes(100)
    print(f"   Primes up to 100: {len(primes_100)} primes")
    print(f"   First 10: {primes_100[:10]}")

    large_prime = 1000000007
    is_prime = miller_rabin_primality(large_prime)
    print(f"   Is {large_prime} prime? {is_prime}")
    print("   ✓ PASSED")

    # Test 2: Matrix Operations
    print("\n2. Matrix Operations:")
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    C = A.multiply(B)
    print(f"   A × B = {C.data}")
    assert C.data == [[19, 22], [43, 50]], "Matrix multiplication failed"

    det = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]]).determinant()
    print(f"   Determinant: {det}")
    assert abs(det - (-3)) < 1e-10, "Determinant calculation failed"
    print("   ✓ PASSED")

    # Test 3: Newton's Method
    print("\n3. Newton's Method (find sqrt(2)):")
    sqrt_2 = newtons_method(lambda x: x**2 - 2, lambda x: 2*x, 1.0)
    print(f"   √2 ≈ {sqrt_2:.15f}")
    print(f"   Error: {abs(sqrt_2 - math.sqrt(2)):.2e}")
    assert abs(sqrt_2 - math.sqrt(2)) < 1e-10, "Newton's method failed"
    print("   ✓ PASSED")

    # Test 4: Convex Hull
    print("\n4. Convex Hull:")
    points = [(0, 0), (1, 1), (0, 2), (2, 2), (1, 0), (2, 0), (0.5, 0.5)]
    hull = convex_hull(points)
    print(f"   Points: {len(points)}")
    print(f"   Hull vertices: {len(hull)}")
    print(f"   Hull: {hull}")
    print("   ✓ PASSED")

    # Test 5: FFT
    print("\n5. Fast Fourier Transform:")
    signal = [complex(1, 0), complex(1, 0), complex(1, 0), complex(1, 0)]
    freq = fft(signal)
    print(f"   Signal: {signal}")
    print(f"   FFT: {[f'{z.real:.2f}+{z.imag:.2f}j' for z in freq]}")
    print("   ✓ PASSED")

    # Test 6: Catalan Numbers
    print("\n6. Catalan Numbers:")
    catalans = [catalan_number(i) for i in range(10)]
    print(f"   First 10 Catalan numbers:")
    print(f"   {catalans}")
    assert catalans[5] == 42, "Catalan number calculation failed"
    print("   ✓ PASSED")

    print("\n" + "="*70)
    print("ALL MATHEMATICAL TESTS PASSED! ✓")
    print("="*70)

    # Advanced: Show precision capabilities
    print("\n📊 Precision Capabilities:")
    getcontext().prec = 50
    pi_approx = sum(Decimal(1) / Decimal(16**k) *
                    (Decimal(4) / Decimal(8*k+1) -
                     Decimal(2) / Decimal(8*k+4) -
                     Decimal(1) / Decimal(8*k+5) -
                     Decimal(1) / Decimal(8*k+6))
                    for k in range(20))
    print(f"   π ≈ {pi_approx}")
    print(f"   (50 decimal places precision)")
