import math
import numbers
import random


def _check_vector(v, name):
    """Raise TypeError or ValueError if v is not a usable vector of real numbers."""
    # Strings and bytes have a length and support indexing, so they reach the
    # multiplication loop and fail there with a confusing message. Reject them up front.
    if isinstance(v, (str, bytes, bytearray)):
        raise TypeError("{} must be a sequence of numbers, not {}".format(name, type(v).__name__))

    # Sets, dicts, generators and scalars cannot be indexed positionally.
    if not hasattr(v, "__len__") or not hasattr(v, "__getitem__"):
        raise TypeError("{} must be an indexable sequence, got {}".format(name, type(v).__name__))

    # Every element must be a real number. Complex values are rejected because the
    # real dot product is not the correct inner product for them.
    for i in range(len(v)):
        value = v[i]
        if isinstance(value, complex) or not isinstance(value, numbers.Real):
            raise TypeError("{}[{}] must be a real number, got {}".format(name, i, type(value).__name__))

        # nan and inf would propagate silently into the result, so refuse them here.
        if not math.isfinite(value):
            raise ValueError("{}[{}] must be finite, got {!r}".format(name, i, value))


# create a function to compute the dot product of two vectors using a for loop
# add comments for the selected function
def dot_product(u, v, validate=True):
    """Return the dot product of two equal-length vectors of real numbers.

    Set validate=False only when the caller has already checked both arguments.
    """
    # Reject unusable types and values before any arithmetic is attempted.
    if validate:
        _check_vector(u, "u")
        _check_vector(v, "v")

    # Both vectors must have the same number of components for the sum to be defined.
    if len(u) != len(v):
        raise ValueError("vectors must have the same length, got {} and {}".format(len(u), len(v)))

    # Accumulate the sum of componentwise products. Neumaier compensated summation
    # tracks the low order bits lost at each addition, so badly scaled inputs such as
    # [1e16, 1, 1, -1e16] do not round away to zero.
    total = 0.0
    compensation = 0.0
    for i in range(len(u)):
        term = u[i] * v[i]
        running = total + term
        if abs(total) >= abs(term):
            compensation += (total - running) + term
        else:
            compensation += (term - running) + total
        total = running
    total += compensation

    # Inputs were all finite, so a non-finite total means the magnitudes overflowed.
    if not math.isfinite(total):
        raise OverflowError("dot product overflowed the range of a float")
    return total


# create a function to compute the matrix-vector product using the dot_product function
# add comments for the selected function
def matvec_multiply(A, x):
    """Return the matrix-vector product A @ x, where A is a sequence of row sequences."""
    # Check x first. Doing this before the empty-matrix shortcut stops a malformed
    # vector from passing unnoticed when A happens to have no rows.
    _check_vector(x, "x")

    # A itself must be an indexable sequence of rows, not a scalar or a generator.
    if isinstance(A, (str, bytes, bytearray)) or not hasattr(A, "__len__") or not hasattr(A, "__getitem__"):
        raise TypeError("A must be an indexable sequence of rows, got {}".format(type(A).__name__))

    # An empty matrix has an empty product, so there is nothing further to compute.
    if len(A) == 0:
        return []

    # Every row must itself be a valid vector as long as x, since each row is dotted
    # against x. Validating here reports the offending row index, and lets the
    # dot_product calls below skip a redundant re-check of x on every row.
    n = len(x)
    for i in range(len(A)):
        row = A[i]
        _check_vector(row, "A[{}]".format(i))
        if len(row) != n:
            raise ValueError("A[{}] has length {}, but x has length {}".format(i, len(row), n))

    # Entry i of the result is the dot product of row i with x.
    result = []
    for row in A:
        result.append(dot_product(row, x, validate=False))
    return result


# create a main function to test the matrix-vector product function using randomly generated data of size 1000x1000
# add comments for the selected function
def main():
    """Test matvec_multiply on random 1000x1000 data against an independent computation."""
    # Fix the seed so the same random test data is produced on every run.
    random.seed(0)

    # Build a random 1000x1000 matrix and a random vector of length 1000.
    n = 1000
    A = [[random.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]
    x = [random.uniform(-1.0, 1.0) for _ in range(n)]

    # Compute the product with the functions under test.
    y = matvec_multiply(A, x)

    # Compute the same product independently, without calling either function,
    # using zip and the built-in sum rather than the indexed loops above.
    expected = [sum(a * b for a, b in zip(row, x)) for row in A]

    # Compare the two results entrywise with a tolerance for floating point rounding.
    tolerance = 1e-9
    max_error = 0.0
    for i in range(n):
        error = abs(y[i] - expected[i])
        if error > max_error:
            max_error = error

    # Report the size of the test, a few sample entries, and whether the check passed.
    print("matrix size: {} x {}".format(len(A), len(A[0])))
    print("vector size: {}".format(len(x)))
    print("result size: {}".format(len(y)))
    print("first 5 computed : {}".format([round(v, 6) for v in y[:5]]))
    print("first 5 expected : {}".format([round(v, 6) for v in expected[:5]]))
    print("max absolute error: {:.3e}".format(max_error))
    print("PASS" if max_error <= tolerance else "FAIL")


if __name__ == "__main__":
    main()
